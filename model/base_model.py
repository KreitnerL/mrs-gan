import os
import pickle
import shutil
import copy
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import scipy.io as io
import torch
from PIL import Image
from tensorboardX import SummaryWriter  # , add_scalar
from util.tensorboard_auxiliary import *
from util.util import mkdirs

__all__ = ['BaseModel']

class BaseModel():
    def name(self, *opt):
        return 'BaseModel'

    def best(self, epoch):
        self.save('best')

    @staticmethod
    def eval_accuracy(estimates, targets):
        assert(targets.size()==estimates.size())
        assert(targets.dim()==estimates.dim())
        assert(targets.device==estimates.device)

        acc = (targets - estimates) / targets
        score = torch.FloatTensor(acc).mean(dim=0)  # size = [1,1,12]
        return {'Cho': score[0], 'Cre': score[1], 'Naa': score[2], 'T2': score[3:5].mean(), 'Noise': score[6],
                'Baseline': score[7:11].mean(), 'Metabolites': score[0:2].mean(), 'Parameters': score[3:-1].mean(),
                'Overall': score.mean()}

    def forward(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        # Define directories
        self.opt.save_dir = os.path.join('./tests/',self.opt.name)
        self.save_dir = self.opt.save_dir
        self.checkpoints_dir = os.path.join(self.save_dir,'checkpoints')
        self.logdir = os.path.join(self.save_dir,'log')
        self.trainimdir = os.path.join(self.save_dir, 'images/training')
        self.lossdir = os.path.join(self.opt.save_dir,'loss')
        self.valdir = os.path.join(self.opt.save_dir,'validation')
        self.debugdir = os.path.join(self.opt.save_dir,'debug')
        mkdirs([self.save_dir, self.checkpoints_dir, self.logdir, self.trainimdir, self.lossdir, self.valdir, self.debugdir])

        self.device = 'cuda' if len(self.opt.gpu_ids)>0 else 'cpu' #self.opt.gpu_ids[0] if len(self.gpu_ids)>0 else None
        # self.log = 0

    def load_network(self, network, network_label, epoch_label, pretrained=False):
        save_filename = '%s_net_%s.pth.tar' % (epoch_label, network_label)
        save_path = os.path.join(self.checkpoints_dir, save_filename)

        if pretrained==True:
            if network_label=='G':
                save_path = './models/pre-trained/' + self.opt.which_model_netG + '.tar'
            elif network_label=='D':
                save_path = './models/pre-trained/' + self.opt.which_model_netD + '.tar'
            elif network_label=='En':
                save_path = './models/pre-trained/' + self.opt.which_model_netEn + '.tar'
            elif network_label=='Est':
                save_path = './models/pre-trained/' + self.opt.which_model_netEst + '.tar'
        else:
            save_filename = '%s_net_%s.pth.tar' % (epoch_label, network_label)
            save_path = os.path.join(self.checkpoints_dir, save_filename)

        network.load_state_dict(torch.load(save_path))

    def optimize(self, epoch=[]):
        pass

    @staticmethod
    def parameter_device_check(net):
        for param in net.parameters():
            if param.device=='cpu':
                print(param.data, param.device)

    def plot_grads(self, arg=True):
        self.opt.plot_grads = arg

    def no_plot_grads(self):
        self.opt.plot_grads = False

    @staticmethod
    def plotall(original, fitted, parameters, estimated, savepth, cropped, ppm, cropRange, individual=True, TB=False):
        Ns, cell_text = 2048, []
        xmin, xmax = 0.0, 5.0
        # ppm = np.linspace(4.7-16, 4.7+16, Ns)
        ppm = ppm[-1::]

        original /= np.max(np.expand_dims(original,axis=0))
        fitted /= np.max(np.expand_dims(fitted,axis=0))
        if cropped:
            ppm = ppm[cropRange[0]:cropRange[1]]
            xmin, xmax = np.min(ppm), np.max(ppm)

        rows = ('Original','Estimated')
        columns = ('Cho','Cre','Naa','Cho t2','Cre t2','Naa t2','Noise','BaseScale0','BaseScale1','BaseScale2','BaseScale3','BaseScale4')
        cell_text.append(['%.5f' % x for x in parameters])
        cell_text.append(['%.5f' % x for x in estimated])

        f = plt.figure()
        ax = f.add_subplot()
        f.suptitle('Original Signal vs Signal from Sampled Parameters')
        ax.plot(ppm, fitted, 'r', label='Estimated')
        ax.plot(ppm, original, label='Original')
        ax.set_xlim(xmax, xmin)
        plt.subplots_adjust(bottom=0.2)

        # Add a table at the bottom of the axes
        table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowLoc='left',
                          colLabels=columns,
                          colLoc='center',
                          loc='bottom')

        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.add_table(table)
        f.canvas.draw()

        path = savepth + '_wTable_axes.pkl'
        file = open(path,'wb')
        pickle.dump(ax, file, protocol=4)
        file.close()

        path = savepth + '_wTable_fig.svg'
        plt.savefig(path, format='svg')

        plt.close('all')

    @staticmethod
    def plotloss(errors, accuracy, x, savedir, label, TB=False):
        LS = np.squeeze(np.asarray(torch.FloatTensor(errors).cpu()))
        AC = np.squeeze(np.asarray(torch.FloatTensor(accuracy).cpu()))
        if LS.ndim == 0:
            LS = np.max(np.expand_dims(LS,axis=0))
        elif LS.ndim >> AC.ndim:
            LS = np.squeeze(LS)
        if AC.ndim == 0:
            AC = np.expand_dims(AC,axis=0)

        # Training Loss
        min = np.min(np.abs(LS)) if not np.isscalar(LS) else LS
        ymin = int(np.argmin(np.abs(LS))) if not np.isscalar(LS) else int(0)
        xmin = x[ymin] if x.size()[0]>>1 else x.item()
        text = "{0:.5f}".format(min)
        y_zero = np.squeeze(np.zeros(x.shape))

        plt.figure()
        plt.xlabel('Epoch')


        if len(errors)>1:
            a = np.asarray(x[0:LS.size])
            b = y_zero[0:LS.size]
            c = LS[0:LS.size]
        else:
            a = np.asarray(x[0])
            b = y_zero
            c = LS

        plt.plot(x[0:LS.size], y_zero[0:LS.size], color='xkcd:silver', linestyle=(0, (5, 10)), zorder=1)
        plt.plot(x[0:LS.size], LS, color='xkcd:dark blue', label='Least Squares Loss', zorder=2)
        if min>0:
            plt.vlines(x=xmin,ymin=0,ymax=min,colors='r',linestyles='dashed')
        else:
            plt.vlines(x=xmin,ymin=min,ymax=0,colors='r',linestyles='dashed')
        plt.plot(xmin,min,'r|',zorder=3)
        plt.plot(xmin,0,'r|')
        plt.ylabel(r'Least Squares Loss')
        plt.annotate(text,xy=(xmin,min), ha='center', verticalalignment='top',xycoords='data')
        plt.annotate('{0:.2f}'.format(xmin),xy=(xmin,0), ha='center', verticalalignment='bottom',xycoords='data')
        # plt.legend(loc=2)
        plt.xlim(left=0)
        plt.title('{} Loss'.format(label))
        plt.xlabel('Epoch')
        # plt.legend(loc=1)
        plt.savefig(savedir+'loss_plots.svg',format='svg')
        plt.savefig(savedir+'loss_plots.png',format='png')

        # Training Error
        plt.figure()
        plt.xlabel('Epoch')
        min = np.min(np.abs(AC)) if not np.isscalar(AC) else AC
        ymin = int(np.argmin(np.abs(AC))) if not np.isscalar(AC) else int(0)
        # print('is x scalar?: ',x, '\n',np.isscalar(x),x.size(),type(x.size()))
        xmin = x[ymin]# if x.size()[0]>>1 else x.item()
        text = "{0:.5f}".format(min)
        # plt.twinx()
        plt.plot(x[0:AC.size],y_zero[0:AC.size], color='xkcd:silver', linestyle=(0, (5, 10)), zorder=1)
        plt.plot(x[0:AC.size],AC,'r',label='Error', zorder=2)
        if np.max(np.abs(AC))>10:
            plt.yscale('symlog')

        if np.min(AC)>0:
            plt.vlines(x=xmin,ymin=0,ymax=np.min(AC),colors='r',linestyles='dashed')
        else:
            plt.vlines(x=xmin,ymin=np.min(AC),ymax=0,colors='r',linestyles='dashed')
        plt.plot(xmin,min,'r|')
        plt.plot(xmin,0,'r|')
        plt.ylabel(r'Error')
        plt.annotate(text,xy=(xmin,min), ha='center', verticalalignment='top',xycoords='data')
        plt.annotate('{0:.2f}'.format(xmin),xy=(xmin,0), ha='center', verticalalignment='bottom',xycoords='data')

        plt.xlim(left=0)
        plt.title('{} Error'.format(label))
        plt.xlabel('Epoch')
        # plt.legend(loc=1)
        plt.savefig(savedir+'error_plots.svg',format='svg')
        plt.savefig(savedir+'error_plots.png',format='png')

        if TB:
            return plt.gcf(), plt.gca()

        plt.close('all')

    @staticmethod
    def plotspectra(original, estimated, savepth, cropped, ppm, cropRange, individual=True, TB=False):
        original = original.cpu()
        estimated = estimated.cpu()
        # ppm = ppm.cpu()
        # cropRange = cropRange.cpu()
        # cropRange = cropRange.cpu()
        # print('>>> Plotting Spectra')
        # print('target.shape: ',original.shape)
        # print('estimate.shape: ', estimated.shape)
        lw, ratio, aspect = 0.25, 1, False
        Ns, sw = 2048, 4000
        # xmin, xmax = 0.0, 5.0
        xmin, xmax = np.min(ppm), np.max(ppm)
        # ppm = np.linspace(4.7-16, 4.7+16, Ns)
        # ppm = ppm[-1::]

        # original /= np.max(np.expand_dims(original,axis=0))
        # estimated /= np.max(np.expand_dims(estimated,axis=0))
        # original /= np.abs(original).max(axis=-1).values
        # estimated /= np.abs(estimated).max(axis=-1).values
        if cropped:
            ppm = np.expand_dims(ppm,axis=0)
            ppm = ppm[0,cropRange[0]:cropRange[1]]
            xmin, xmax = np.min(ppm), np.max(ppm)

        f, ax = plt.subplots(1,1)
        ax.plot(ppm, estimated, 'r', label='Estimated')
        ax.plot(ppm, original, label='Original') #'-.',color='cornflowerblue', # t,
        ax.set_xlim(xmax, xmin)
        ax.set(title='Original Signal vs Signal from Sampled Parameters')
        ax.set_aspect(1/ax.get_data_ratio()*ratio)

        f.tight_layout(h_pad=1)

        path = savepth + '_axes.pkl'
        file = open(path,'wb')
        pickle.dump(ax, file, protocol=4)
        file.close()

        path = savepth + '_fig.svg'
        plt.savefig(path, format='svg')

        path = savepth + '_fig.png'
        plt.savefig(path, format='png')
        plt.close('all')

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def save(self, label):
        pass

    def save_summary(self, label=None):
        save_filename = 'summary.pkl'
        save_path = os.path.join(self.lossdir, save_filename)
        data = copy.deepcopy(self.summary())
        for K, V in data.items():
            if V.__class__.__name__=='dict':
                for k, v in data[K].items():
                    try:
                        data[K][k] = v.cpu()
                    except AttributeError:
                        pass
            else:
                try:
                    data[K] = V.cpu()
                except AttributeError:
                    pass
        with open(save_path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        del data

    def save_network(self, network, epoch_label, gpu_ids):
        save_filename = '%s_net.pth.tar' % (epoch_label)
        save_path = os.path.join(self.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids)>0 and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def set_input(self, input):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def update_learning_rate(self, **kwargs):
        pass

    # def wlog(self, phase, epoch, lr):
    #     # assert(len(lr)==2)
    #     # glr, dlr = lr[0], lr[1]
    #     # while self.log==0:
    #     # global writer
    #     # writer = SummaryWriter(os.path.join(self.opt.save_dir,'log'))
    #     #     self.log = 1
    #     # if self.log==1:
    #     if phase=='train':
    #         writer = SummaryWriter(os.path.join(self.opt.save_dir,'log/train/'))#+ datetime.now().strftime("%Y%m%d-%H%M%S")))
    #         save_filename = 'Epoch_%s_%s_errors.pkl' % (epoch, 'training')
    #         save_path = os.path.join(self.logdir, save_filename)
    #         if os.path.isfile(save_path):
    #             with open(save_path, 'rb') as file:
    #                 unpickler = pickle.Unpickler(file)
    #                 dict = unpickler.load()
    #                 for i in range(len(dict['ls_loss'])):
    #                     writer.add_scalar('Train/loss', dict['ls_loss'][i], i)
    #                     # writer.add_scalar('Train/Loss_generator', dict['G'][i], i)
    #                     # writer.add_scalar('Train/wasserstein_distance', dict['WD'][i], i)
    #                     # writer.add_scalar('Training/Feature_loss', dict['Feat'][i], i)
    #                 writer.add_scalar('Train/lr', lr, epoch)
    #                 # writer.add_scalar('Train/generator_lr', glr, epoch)
    #                 # Todo: Fix this plot function
    #                 dx = 1 / (self.dataset_sizes[0] / (self.opt.print_freq / 2))
    #                 x = np.arange(start=0,stop=self.n_epochs,step=dx)
    #                 F, AX = self.plotloss(self.error_dict, self.accuracy, x, os.path.join(self.lossdir,'dynamic_training_plots'), TB=True)
    #                 writer.add_image("Training Spectra: Epoch {}".format(epoch), plot_to_image(F))
    #         writer.close()
    #     elif phase=='val':
    #         writer = SummaryWriter(os.path.join(self.opt.save_dir,'log/val/'))#+ datetime.now().strftime("%Y%m%d-%H%M%S")))
    #         save_filename = 'Epoch_%s_%s_errors.pkl' % (epoch, 'validation')
    #         save_path = os.path.join(self.logdir, save_filename)
    #         if os.path.isfile(save_path):
    #             with open(save_path, 'rb') as file:
    #                 unpickler = pickle.Unpickler(file)
    #                 dict = unpickler.load()
    #                 for i in range(len(dict['ls_loss'])):
    #                     self.writer.add_scalar('Validation/loss', dict['ls_loss'][i], i)
    #                     # self.writer.add_scalar('Validation/Loss_generator', dict['G'][i], i)
    #                     # self.writer.add_scalar('Validation/discriminator_accuracy', dict['accuracy'][i], i)
    #                     # self.writer.add_scalar('Validation/wasserstein_distance', dict['WD'][i], i)
    #                 self.writer.add_scalar('Validation/lr', lr, epoch)
    #                 # self.writer.add_scalar('Validation/generator_lr', glr, epoch)
    #
    #     # if epoch==(self.opt.niter + self.opt.niter_decay + 1):
    #         writer.close()

    def zipFold(self):
        shutil.make_archive(self.save_dir, 'zip', self.save_dir)
        if os.path.isdir(self.save_dir + '.zip') or os.path.isfile(self.save_dir + '.zip'):
            shutil.rmtree(self.save_dir)
        else:
            print('{} not found. Directory not removed.'.format(self.save_dir + '.zip'))

    def collect_images(self):
        fileslist = os.listdir(self.lossdir)
        images = OrderedDict({})
        for file in fileslist:
            if file.endswith('.png'):
                base = os.path.basename(file)
                if 'dynamic' in base:
                    images.update({base: (np.asarray(Image.open(file)), file)})
        fileslist = os.listdir(self.trainimdir)
        for file in fileslist[-1::]:
            if file.endswith('.png'):
                base = os.path.basename(file)
                if 'epoch' in base:
                    images.update({base: (np.asarray(Image.open(file)), file)})

