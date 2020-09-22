# NOT USED
# Example implementation of a network using define.py

import copy
import math
import os

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import util.util as util
# from torch.distributions.normal import Normal
# import matplotlib.pyplot as plt
from model.aux.auxiliary import plot_grad_flow, print_network
from model.base_model import BaseModel
# from data.data_auxiliary import normalizeSpectra
from model.define import define
from modules.metrics import (mean_squared_error,
                             pearson_correlation_coefficient, percent_error)
from modules.updated_physics_modelv2 import PhysicsModelv2 as PhysicsModel
# from modules.learned_group_modules import stretch_model
from types import SimpleNamespace

__all__ = ['FittingModel']

# python /home/john/SpectroscopyModel/train_condensed.py --dataroot '/home/john/SpectroscopyModel/dataset' --model 'fitting' --file_ext '.mat' --name 'Phase01_test_010' --phase 'train' --save_latest_freq 28000 --print_freq 14000 --no_flip --batchSize 500 --checkpoints_dir './tests' --niter 75 --niter_decay 0 --quiet --normalize --magnitude --actvn 'relu' --cropped_signal --lr 0.0001 --G1 --plot_grads --dataset_mode 'LabeledMatSpectralDataset' --which_model_netEst 'resnet50' --parameters --metabolites --lambda_metab 1 --test_split 0 --phase_data_path './dataset/'

class FittingModel(BaseModel):
    def name(self):
        return 'FittingModel'

    def initialize(self, opt):
        self.counter = 0
        BaseModel.initialize(self, opt)
        self.dataset_sizes = torch.tensor([np.genfromtxt('/home/john/SpectroscopyModel/dataset/sizes.txt',delimiter=',').astype(np.float32)]).squeeze()
        self.n_epochs = opt.niter + opt.niter_decay
        self.lr = [[opt.lr, 0]]
        self.epoch = -1
        # if opt.plot_grads:
        #     self.ave_grads, self.layers = [], []

        # Initialize LCM Physics Model
        self.cropped = opt.cropped_signal
        self.PhysicsModel = PhysicsModel(cropped=self.cropped, magnitude=opt.magnitude, range=opt.cropRange, gpu_ids=opt.gpu_ids)#.cuda_(self.device)
        self.PhysicsModel = self.PhysicsModel.to(self.device)
        self.cropRange = opt.cropRange


        # Initialize summary dictionary
        dictionary, self.index = self.PhysicsModel.initialize()
        self._summary = SimpleNamespace(**{'training': {
                                               'ls_loss': [],
                                               'mse': copy.deepcopy(dictionary),
                                               'r2': copy.deepcopy(dictionary),
                                               'error': copy.deepcopy(dictionary),
                                               'xaxis': None
                                           },
                                           'validation': {
                                               'ls_loss': [],
                                               'mse': copy.deepcopy(dictionary),
                                               'r2': copy.deepcopy(dictionary),
                                               'error': copy.deepcopy(dictionary),
                                               'xaxis': None
                                           },
                                           'temp': {
                                               'ls_loss': [],
                                               'mse': copy.deepcopy(dictionary),
                                               'r2': copy.deepcopy(dictionary),
                                               'error': copy.deepcopy(dictionary)
                                           },
                                           'data': {'num_training_spectra': int(self.dataset_sizes[0]),
                                                    'num_training_batches': int(torch.ceil(self.dataset_sizes[0]/torch.tensor(opt.batchSize,dtype=torch.float32))),
                                                    'num_validation_spectra': int(self.dataset_sizes[1]),
                                                    'num_validation_batches': int(torch.ceil(self.dataset_sizes[1]/torch.tensor(opt.batchSize,dtype=torch.float32)))},
                                           'lr': self.lr
        })

        dx = 1 / (self.dataset_sizes[0] / (self.opt.print_freq))
        self._summary.training['xaxis'] = torch.arange(start=0,end=self.n_epochs,step=dx)
        self._summary.validation['xaxis'] = torch.arange(start=1,end=self.n_epochs+1,step=1)
        del dx

        # Define the Network
        inc = 2 if not self.opt.magnitude else 1
        self.netEst = define.Estimator(dim=1, input_nc=inc, output_nc=23, ngf=64, padding_type=opt.pad, actvn_type=opt.actvn,
                                       norm_type=opt.norm, n_blocks=opt.n_blocks, gpu_ids=self.gpu_ids, se=opt.se, n_downsampling=2,
                                       pAct=opt.pAct, use_dropout=opt.use_dropout, depth=opt.depth, use_sigmoid=opt.use_sigmoid,
                                       which_model_netEst=opt.which_model_netEst, learned_g=opt.learned_g, num_down=opt.num_down,
                                       growth_rate=opt.growth_rate, flexible=opt.flexible)

        # def nan_hook(self, inp, output):
        #     if not isinstance(output, tuple):
        #         outputs = [output]
        #     else:
        #         outputs = output
        #
        #     # print('Input.shape: ', inp.shape)
        #     # print('Output.shape: ', output.shape)
        #
        #     for i, out in enumerate(outputs):
        #         print('Output {} size: {}'.format(i,out.shape))
        #         print('Output: ', out)
        #         nan_mask = torch.isnan(out)
        #         if nan_mask.any():
        #             print("In", self.__class__.__name__)
        #             raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
        #
        # for submodule in self.netEst.modules():
        #     submodule.register_forward_hook(nan_hook)



        if self.opt.isTrain or opt.continue_train:
            self.netEst.train()
        else:
            self.netEst.eval()

        if self.opt.isTrain and opt.continue_train:
            self.netEst.train()
            which_epoch = opt.which_epoch
            self.load_network(self.netEst,'Est',which_epoch)

        if not self.opt.isTrain:
            which_epoch = opt.which_epoch
            self.load_network(self.netEst,'Est',which_epoch)
            self.netEst.eval()

        if self.isTrain:
            # self.criterionIdt = nn.L1Loss()
            self.criterionMSE = nn.MSELoss()#mean instead of sum # reduction='sum')     # Least-Squares Loss - L2 summed
            self.optimizer = torch.optim.Adam(self.netEst.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        print('------------ Networks initialized ------------')
        self.num_params = 0
        print_network(self.netEst)

        # Save to the disk
        file_name = os.path.join(self.opt.checkpoints_dir, 'model_architecture.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------- Architecture --------------\n')
            opt_file.write('-------------  Estimator   --------------\n')
            opt_file.write(print_network(self.netEst, file=True))
            opt_file.write('----------------- End -------------------\n')
        print('----------------------------------------------')


    def eval(self):
        self.netEst.eval()

    def train(self):
        self.netEst.train()

    def set_input(self, input):
        if self.opt.magnitude:
            self.data = input['magnitude'].to(self.device)
        else:
            self.data = input['spectra'].to(self.device)
        self.data = self.data if not self.cropped else self.data[:,:,self.opt.cropRange[0]:self.opt.cropRange[1]]
        self.target = input['params'].to(self.device)

    @property
    def ppm(self):
        return getattr(self.PhysicsModel,'ppm')

    def forward(self):
        torch.autograd.set_detect_anomaly(True)
        self.parameters = self.netEst(self.data)
        self.parameters = self.parameters.unsqueeze(-1)

    def quantify(self):
        return self.PhysicsModel.quantify(self.parameters)

    def backward(self, epoch=[], back=True):
        torch.autograd.set_detect_anomaly(True)
        self.loss = torch.zeros(1).to(self.parameters.device)

        self.loss_params = self.criterionMSE(self.parameters,self.target)
        self.loss += self.loss_params
 
        if self.opt.metabolites:
            self.loss_metab = self.criterionMSE(self.parameters[:,self.index[-3]],self.target[:,self.index[-3]])
            self.loss += self.loss_metab * self.opt.lambda_metab
        
        if self.opt.parameters:
            self.loss_param = self.criterionMSE(self.parameters[:,self.index[-2]],self.target[:,self.index[-2]])
            self.loss += self.loss_param#.to(self.parameters.device)

        if self.opt.phase_loss:
            self.loss_phase = self.criterionMSE(self.parameters[:,self.index[11]],self.target[:,self.index[11]])
            self.loss += self.loss_phase * self.opt.lambda_phase

        if back: self.loss.backward()

        if self.opt.plot_grads:
            plot_grad_flow(self.netEst, self.debugdir, epoch)

    def optimize(self, back=True, epoch=[]):
        if len(self.gpu_ids)>0:
            assert(torch.cuda.is_available())

        self.forward()

        self.optimizer.zero_grad()
        self.backward(epoch, back=back)
        self.optimizer.step()

    def evaluate(self, target, estimate, dict, ind):
        target_quant = self.PhysicsModel.quantify(target.detach())
        estimate_quant = self.PhysicsModel.quantify(estimate.detach())
        keys = dict['mse'].keys()

        dict['ls_loss'].append(self.loss.detach())
        for i, k in zip(ind, keys):
            if i.__class__.__name__ == 'int':
                dict['mse'][k].append(mean_squared_error(target[::,i].detach(), estimate[::,i].detach()))
                dict['error'][k].append(percent_error(target_quant[::,i].detach(), estimate_quant[::,i].detach()))
                a = pearson_correlation_coefficient(target[:,i,:].detach(), estimate[:,i,:].detach())
                dict['r2'][k].append(a)
            elif i.__class__.__name__ == 'tuple':
                temp1, temp2, temp3 = [], [], []
                for ii in i:
                    temp1.append(mean_squared_error(target[::,ii].detach(), estimate[::,ii].detach()))
                    temp2.append(percent_error(target_quant[::,ii].detach(), estimate_quant[::,ii].detach()))
                    arg = pearson_correlation_coefficient(target[::,ii].detach(),estimate[::,ii].detach())
                    if not torch.isnan(arg): temp3.append(arg)#target[::,ii].detach(), estimate[::,ii].detach()))
                temp1 = torch.FloatTensor(temp1).mean()
                temp2 = torch.FloatTensor(temp2).mean()
                temp3 = torch.FloatTensor(temp3).mean()

                dict['mse'][k].append(temp1)
                dict['error'][k].append(temp2)
                dict['r2'][k].append(temp3)

    def collect_train_errors(self):
        self.evaluate(self.parameters, self.target, self._summary.training, self.index)

    def collect_val_errors(self):
        self.evaluate(self.parameters, self.target, self._summary.temp, self.index)

    def compile_val(self):
        self.epoch += 1
        for key in self._summary.temp.keys():
            if key=='ls_loss':
                self._summary.validation[key].append(torch.mean(torch.FloatTensor([self._summary.temp[key]]).clone()))
                self._summary.temp[key] = []
            else:
                for k in self._summary.temp[key].keys():
                    self._summary.validation[key][k].append(torch.mean(torch.FloatTensor([self._summary.temp[key][k]]).clone()))
                    self._summary.temp[key][k] = []
        return torch.FloatTensor([self._summary.validation['mse']['Overall'][-1]])

    def summary(self):
        return {'training': self._summary.training, 'validation': self._summary.validation,
                'index': self.index, 'lr': self._summary.lr, 'epoch': self.epoch,
                'data': self._summary.data}

    def save(self, *epoch):
        self.save_network(self.netEst, epoch, self.gpu_ids)

    def plot_loss(self):
        self.plotloss(self._summary.training['ls_loss'], self._summary.training['error']['Overall'], self._summary.training['xaxis'],#.unsqueeze(-1),
                      os.path.join(self.lossdir,'dynamic_training_'), 'Training')

    def plot_val(self):
        self.plotloss(self._summary.validation['ls_loss'], self._summary.validation['error']['Overall'], self._summary.validation['xaxis'],#.unsqueeze(-1),
                      os.path.join(self.lossdir,'dynamic_validation_'), 'Validation')

    def save_spectra(self, epoch, batch):
       fitted = self.PhysicsModel(self.parameters.detach().cpu())
        fitted = fitted.cpu()
        for i in range(batch):
            if batch >= 0:
                path = os.path.join(self.trainimdir,'epoch_{}_batch_{}'.format(epoch, i))
                self.plotspectra(original=self.data[i,0,:].cpu(), estimated=fitted[i,0,:], savepth=path, cropped=self.cropped, ppm=np.asarray(self.ppm.cpu()), cropRange=self.opt.cropRange)
            else:
                path = os.path.join(self.trainimdir,'epoch_{}'.format(epoch))
                self.plotspectra(original=self.data[0,0,:].cpu(), estimated=fitted[0,0,:], savepth=path, cropped=self.cropped, ppm=np.asarray(self.ppm.cpu()), cropRange=self.opt.cropRange)

        io.savemat(os.path.join(self.trainimdir,'epoch_{}.mat'.format(epoch)), do_compression=True,
                   mdict={'estimates': np.asarray(self.parameters.cpu().detach()),
                          'reconstructed': np.asarray(fitted.cpu().detach()),
                          'original': np.asarray(self.data.cpu().detach()),
                          'targets': np.asarray(self.target.cpu().detach())})
        del fitted

    def reconstruct(self, params):
        return self.PhysicsModel(params)

    def warmup(self, nBatch):
        self.nBatch = nBatch
        if self.opt.warmup != None:
            lr = self.lr[0][0] / (nBatch)# * self.opt.warmup)
            self.lr.append([lr,0])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def update_learning_rate(self, epoch, batch, n=10, **kwargs): # Cycle, non-cycle: TTUR, no_TTUR
        self.epoch = epoch
        if self.opt.warmup != None and epoch <= self.opt.warmup:
            lr = (batch + 1) * self.lr[1][0] / self.opt.warmup
            num = (epoch - self.opt.starting_epoch) + (batch / self.nBatch)
        elif self.opt.lr_method == 'cosine':
            T_total = self.n_epochs * self.nBatch
            T_cur = (epoch % self.n_epochs) * self.nBatch + batch if self.opt.warmup==None else (epoch % (self.n_epochs - self.opt.warmup)) * self.nBatch + batch
            lr = 0.5 * self.lr[0][0] * (1 + math.cos(math.pi * T_cur / T_total))
            num = (epoch - self.opt.starting_epoch) + (batch / self.nBatch)
        elif self.opt.lr_method == None and self.opt.niter_decay > 0:
            if epoch <= self.opt.niter:
                lr = None
            else:
                lrd = self.lr[0][0] / self.opt.niter_decay
                lr = self.lr[-1][0] - lrd
                num = epoch
        # elif self.opt.lr_method == None and self.opt.niter_decay > 0:
        #     lrd = self.lr[0] / self.opt.niter_decay
        #     lr = self.lr[-1] - lrd
        else:
            lr = None
            # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            # lr = self.lr[0] * (0.1 ** (epoch // n))

        if not lr==None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr.append([lr, num])
            self._summary.lr = self.lr
            # print('self.lr[-2]: ',self.lr[-2][0])
            # print('lr: ',lr)
            # print('update learning rate: %f -> %f' % (torch.FloatTensor([self.lr[-2][0]]), torch.FloatTensor([lr])))
