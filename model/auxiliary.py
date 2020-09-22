# from modules.generator_networks import ResnetGenerator
# from modules.discriminator_networks import *
# from modules.loss_networks import *
import os
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import itertools
import matplotlib.pyplot as plt
from types import SimpleNamespace
from modules.SNlayers import *


###############################################################################
# Functions
###############################################################################
# Todo: fix the weights_init to account for different dimensionalities - done!
def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1
    # init
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        # init.kaiming_normal_(m.weight.data)#, mean=1, std=0.02)
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)  # todo: why is the bias set to 0?
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        # init.normal_(m.bias)#.data)
        # init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            init.orthogonal_(param.data) if len(param.shape) >= 2 else init.normal_(param.data)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.GRUCell):
        for param in m.parameters():
            init.orthogonal_(param.data) if len(param.shape) >= 2 else init.normal_(param.data)
    # elif isinstance(m, nn.utils.spectral_norm):
    #     init.kaiming_normal_(m.weight.data)
    #     init.kaiming_normal_(m.bias.data)

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def get_norm_layer(dim, norm, arg):
    assert(dim==1 or dim==2 or dim==3)
    norm_type = []
    if dim==1:
        if norm=='instance':
            return [nn.InstanceNorm1d(arg)]
        elif norm=='batch':
            return [nn.BatchNorm1d(arg)]
        elif norm=='spectral':
            norm_layer = torch.nn.utils.spectral_norm(arg[0])
            norm_layer.__setattr__('name','SpectralNorm(Conv1d)')
            return [norm_layer]
        elif 'spectral' in norm and 'mean' in norm:
            norm_layer = [MeanSpectralNorm1d(arg)]
            norm_layer.__setattr__('name','MeanSpectralNorm1d')
            return [norm_layer]
    elif dim==2:
        if norm=='instance':
            return [nn.InstanceNorm2d(arg)]
        elif norm=='batch':
            return [nn.BatchNorm2d(arg)]
        elif norm=='spectral':
            norm_layer = torch.nn.utils.spectral_norm(arg[0])
            norm_layer.__setattr__('name','SpectralNorm(Conv2d)')
            return [norm_layer]
        elif 'spectral' in norm and 'mean' in norm:
            norm_layer = [MeanSpectralNorm2d(arg)]
            norm_layer.__setattr__('name','MeanSpectralNorm2d')
            return [norm_layer]
    elif dim==3:
        if norm=='instance':
            return [nn.InstanceNorm3d(arg)]
        elif norm=='batch':
            return [nn.BatchNorm3d(arg)]
        elif 'spectral' in norm and 'mean' in norm:
            norm_layer = [MeanSpectralNorm3d(arg)]
            norm_layer.__setattr__('name','MeanSpectralNorm3d')
            return [norm_layer]
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")

def get_conv_layer(dim, net, in_c, out_c, kernel, groups=1, stride=1, pad=0, pad_o=0, bias=False):#*norm):#type(dim, *norm):#, *eq):
    in_c=int(in_c)
    out_c=int(out_c)
    if dim==1 and net=='conv':
        return [nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=bias, groups=groups)]
    elif dim==1 and net=='trans':
        return [nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, output_padding=pad_o, bias=bias, groups=groups)]
    elif dim==2 and net=='conv':
        return [nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=bias, groups=groups)]
    elif dim==2 and net=='trans':
        return [nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, output_padding=pad_o, bias=bias, groups=groups)]
    elif dim==3 and net=='conv':
        return [nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=bias, groups=groups)]
    elif dim==3 and net=='trans':
        return [nn.ConvTranspose3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, output_padding=pad_o, bias=bias, groups=groups)]
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")

def get_conv_type(dim, net):
    if dim==1 and net=='conv':
        return 'Conv1d'
    elif dim==1 and net=='trans':
        return 'ConvTranspose1d'
    elif dim==2 and net=='conv':
        return 'Conv2d'
    elif dim==2 and net=='trans':
        return 'ConvTranspose2d'
    elif dim==3 and net=='conv':
        return 'Conv3d'
    elif dim==3 and net=='trans':
        return 'ConvTranspose3d'
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")

def get_actvn_layer(actvn='relu'):
    if actvn=='relu':
        return [nn.ReLU(True)]
    elif actvn=='tanh':
        return [nn.Tanh()]
    elif actvn=='leakyrelu':
        return [nn.LeakyReLU(0.2, True)]
    elif actvn=='prelu':
        return [nn.PReLU(num_parameters=1, init=0.25)]
    elif actvn=='rrelu':
        return [nn.RReLU(inplace=True)]
    elif actvn=='selu':
        return [nn.SELU(inplace=True)]
    elif actvn=='none':
        return None
    else:
        raise ValueError("Activation function {} not found".format(actvn))

def get_padding_layer(dim, padding='reflect', arg=1, *block):
    # padding_layer = []
    if dim==1:
        # arg = [1, 1, 0]
        if 'reflect' in padding:# == 'reflect':
            # print('type(get_padding_layer)', type(nn.ReflectionPad1d([arg, arg])))
            return [nn.ReflectionPad1d([arg, arg])]
        elif 'replicat' in padding:# == 'replicate':
            # print('type(get_padding_layer)', type(nn.ReplicationPad1d(arg)))
            return [nn.ReplicationPad1d(arg)]
        elif padding == 'zero':
            return [nn.ConstantPad1d(arg, 0)]
    elif dim==2:
        if padding == 'reflect':
            return [nn.ReflectionPad2d(arg)]
        elif padding == 'replicate':
            return [nn.ReplicationPad2d(arg)]
        elif padding == 'zero':
            return [nn.ConstantPad2d(arg, 0)]
    elif dim==3:
        if padding == 'replicate':
            return [nn.ReplicationPad3d(arg)]
        elif padding == 'zero':
            return [nn.ConstantPad3d(arg, 0)]
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")
    # return padding_layer

def get_adaptive_pooling_layer(dim, type, size):#*norm):#type(dim, *norm):#, *eq):
    if dim==1 and 'avg' in type:
        return [nn.AdaptiveAvgPool1d(size)]
    elif dim==1 and 'max' in type:
        return [nn.AdaptiveMaxPool1d(size)]
    elif dim==2 and 'avg' in type:
        return [nn.AdaptiveAvgPool2d(size)]
    elif dim==2 and 'max' in type:
        return [nn.AdaptiveMaxPool2d(size)]
    elif dim==3 and 'avg' in type:
        return [nn.AdaptiveAvgPool3d(size)]
    elif dim==3 and 'max' in type:
        return [nn.AdaptiveMaxPool3d(size)]
    else:
        raise ValueError("Input data dimensionality must be between 1D and 3D")

def create_optimizer(network, opt, type=None, method='Adam'):#, add_params):
    # Defining Network Parameters
    # Generator and optional Encoder Networks
    if not type=='AE':
        params = list(network.parameters())
    else:
        params = itertools.chain(network[0].parameters(), network[1].parameters())

    # Define the Learning Rates
    # Two Time-scale Update Rule is used by default
    if opt.no_TTUR or type=='AE':
        if opt.beta1 and opt.beta2:
            beta1, beta2 = opt.beta1, opt.beta2
        else:
            beta1, beta2 = opt.beta, opt.beta
        lr = opt.lr
    else:
        beta1 = opt.beta1 if opt.beta1 else 0
        beta2 = opt.beta2 if opt.beta2 else 0.9

    if type=='gen' or type=='dis':
        lr = opt.glr / (opt.timestep_ratio / 2) if type=='gen' else opt.dlr * (opt.timestep_ratio / 2)
    else:
        lr = opt.lr

    # Define the Optimizers
    if method=='Adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
    elif method=='RMS':
        optimizer = torch.optim.RMSprop(params, lr=lr)
    elif method=='SGD':
        optimizer = torch.optim.SGD(params, lr=lr)#, momentum, dampening, weight_decay)

    return optimizer

def print_network(net, file=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # num_params += num_params
    if file==False:
        print(net)
        print('Total number of parameters: %d' % num_params)
    else:
        string = '\n'.join([str(net), str('Total number of parameters: %d' % num_params), str('\n')])
        return string


def progressbar(it, prefix="", size=50, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def accuracy(y_true, y_pred, normalize=True):
    acc = []
    assert(len(y_true)==len(y_pred))
    assert(y_true.dim()==y_pred.dim())
    assert(y_true.device==y_pred.device)

    if y_true.dim() > 1:
        y_true = torch.squeeze(y_true)
    # if y_pred.dim() > 1:
        y_pred = torch.squeeze(y_pred).detach()

    for i in range(len(y_true)):
        acc.append((y_true[i] - y_pred[i])/y_true[i])

    if normalize==True:
        score = torch.FloatTensor(acc).mean()
    else:
        score = (torch.FloatTensor(acc)>=0.5).nonzero().numel()

    return score

def amean(input):
    d = np.asarray(input).ndim
    out = input
    for i in range(d):
        out = np.mean(out)
    return out

# Code from RoshanRane
# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
# def collect_grad_flow(model, ave_grads=[], layers=[]):
#     for n, p in model.named_parameters():
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#
# def plot_grad_flow(model, ave_grads, layers, savedir, *epoch):
#     plt.plot(ave_grads, alpha=0.3, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation='vertical')#45)
#     plt.xlim(xmin=0, xmax=len(ave_grads))
#     plt.xlabel("Layers")
#     plt.ylabel("Average Gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.yscale('log')
#     # plt.show()
#     plt.axes().set_aspect('auto')
#     # asprt = ratio /
#     if epoch:
#         name = model.Function + '_epoch_' + str(epoch)
#     else:
#         name = model.Function
#     path = os.path.join(savedir,name)
#     plt.savefig(path + '.svg',format='svg', bbox_inches='tight')
#     plt.savefig(path + '.png',format='png', bbox_inches='tight')
#     # plt.close()

def plot_grad_flow(model, savedir, *epoch):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation='vertical')#45)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    # plt.show()
    plt.axes().set_aspect('auto')
    # asprt = ratio /
    if epoch:
        name = model.Function + '_epoch_' + str(epoch)
    else:
        name = model.Function
    path = os.path.join(savedir,name)
    plt.savefig(path + '.svg',format='svg', bbox_inches='tight')
    plt.savefig(path + '.png',format='png', bbox_inches='tight')
    # plt.close()

import sys
import requests

def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

# def progressbar_loading(dir, prefix="", size=50, func, path, file=sys.stdout):
#     loaded = 0
#     total = int(sum(os.path.getsize(f) for f in os.listdir(dir) if os.path.isfile(f))/1000)
#     # for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
#     loaded += len(data)
#     done = int(50*downloaded/total)
#     show(0)
#     for i, item in enumerate(dir):
#         yield item
#         file.write('{}[{}{}] {}/{}\r'.format(prefix, '#' * done, '.' * (size-done), ))
#         file.flush()
#     file.write('\n')
#     file.flush()

# find the size of data to be loaded
# monitor the io.read speed and update the amount of data read based on time elapsed and read rate

def convertdict(file, simple=False, device='cpu'):
    if simple:
        p = SimpleNamespace(**file) # self.basisSpectra
        keys = [y for y in dir(p) if not y.startswith('__')]
        for i in range(len(keys)):
            file[keys[i]] = torch.FloatTensor(np.asarray(file[keys[i]], dtype=np.float64)).squeeze().to(device)
        return SimpleNamespace(**file)
    else:
        delete = []
        for k, v in file.items():
            if not k.startswith('__'):
                file[k] = torch.FloatTensor(np.asarray(file[k], dtype=np.float64)).squeeze().to(device)
            else:
                delete.append(k)
        if len(delete)>0:
            for k in delete:
                file.pop(k, None)
        return file

# def convertdict(file, simple=False):
#     if simple:
#         p = SimpleNamespace(**file) # self.basisSpectra
#         keys = [y for y in dir(p) if not y.startswith('__')]
#         for i in range(len(keys)):
#             file[keys[i]] = torch.FloatTensor(np.asarray(file[keys[i]], dtype=np.float64)).squeeze()
#         return SimpleNamespace(**file)
#     else:
#         delete = []
#         for k, v in file.items():
#             if not k.startswith('__'):
#                 file[k] = torch.FloatTensor(np.asarray(file[k], dtype=np.float64)).squeeze()
#             else:
#                 delete.append(k)
#         if len(delete)>0:
#             for k in delete:
#                 file.pop(k, None)
#         return file


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss



#
from statistics import mean
def correlation_coefficient(target, estimate):
    '''
    Code adapted from: https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial/

    :param target:
    :param estimate:
    :return: R^2 value
    '''
    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
             ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

    def squared_error(ys_orig, ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))

    def coefficient_of_determination(ys_orig, ys_line):
        y_mean_line = [mean(ys_orig) for y in ys_orig]
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr/squared_error_y_mean)

    m, b = best_fit_slope_and_intercept(estimate, target)
    regression_line = [(m*x)+b for x in estimate]

    return coefficient_of_determination(target, regression_line)


def matrix_inverse(matrix):
    s = matrix.shape
    if len(s)==2:
        if s[0] < s[1]:
            return right_inverse(matrix)
        elif s[0] > s[1]:
            return left_inverse(matrix)
        else:
            return torch.inverse(matrix)
    # elif len(s) == 3:
    #     if s[1] < s[2]:
    #         return right_inverse(matrix)
    #     elif s[1] > s[2]:
    #         return left_inverse(matrix)
    #     else:
    #         return torch.inverse(matrix)

def left_inverse(matrix):
    return torch.matmul(torch.inverse(torch.matmul(matrix.t(),matrix)),matrix.t())

def right_inverse(matrix):
    return torch.matmul(matrix.t(), torch.inverse(torch.matmul(matrix,matrix.t())))

def conditional_loss(criterion, min, max, input):
    loss = torch.empty([input.shape[0],1])
    for n in input.shape[0]:
        for m in input.shape[1]:
            if input[n,m,:] << min:
                loss[n,0] = criterion(input,min)
            elif input[n,m,:] >> max:
                loss[n,0] = criterion(input,max)
            else:
                loss[n,0] = criterion(input,input)
    return loss



def mAIC(estimate, target, ED, m=5):
    crit = nn.MSELoss()
    return torch.log(crit(estimate, target)) + 2*m * ED / target[0,0,:].shape



import pickle
import io
import scipy.io as sio
# https://github.com/pytorch/pytorch/issues/16797 - from mmcdermott
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def pkl2mat(input,save):
    with(input, 'rb') as file:
        dict = CPU_Unpickler(file).load()
    sio.savemat(save,mdict=dict,do_compression=True)


def spectral_norm(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                del(child)
            elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                model._modules[m] = nn.utils.spectral_norm(child)
                del(child)
        else:
            spectral_norm(child) # Used to convert submodules
    return model

from modules.SNlayers import MeanSpectralNorm1d, MeanSpectralNorm2d, MeanSpectralNorm3d
'''
Mean  Spectral  Normalization  of  Deep  Neural  Networks  for  EmbeddedAutomation, 2019 IEEE CASE
By Anand Krishnamoorthy Subramanian1 and Nak Young Chong1
'''
def mean_spectral_norm(model):
    for m in model._modules:
        child = model._modules[m]
        if is_leaf(child):
            if isinstance(child, nn.BatchNorm1d):
                model._modules[m] = MeanSpectralNorm1d(child.num_features)
                del(child)
            if isinstance(child, nn.BatchNorm2d):
                model._modules[m] = MeanSpectralNorm2d(child.num_features)
                del(child)
            if isinstance(child, nn.BatchNorm3d):
                model._modules[m] = MeanSpectralNorm3d(child.num_features)
                del(child)
        else:
            mean_spectral_norm(child) # Used to convert submodules
    return model

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_num_gen(gen):
    return sum(1 for x in gen)
