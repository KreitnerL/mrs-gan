# Implementation of all basic functionality a model should have.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizers = dict()
        self.schedulers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def optimize_parameters(self, optimize_G=True, optimize_D=True):
        pass

    def plot_grads(self, arg=True):
        self.opt.plot_grads = arg

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_losses(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = {}
        for name, optimizer in self.optimizers.items():
            old_lr[name] = optimizer.param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        for name, optimizer in self.optimizers.items():
            print(name, ': learning rate %.7f -> %.7f' % (old_lr[name], optimizer.param_groups[0]['lr']))

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

        lw, ratio, aspect = 0.25, 1, False
        Ns, sw = 2048, 4000
        # xmin, xmax = 0.0, 5.0
        xmin, xmax = np.min(ppm), np.max(ppm)

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

        # def zipFold(self):
        # shutil.make_archive(self.save_dir, 'zip', self.save_dir)
        # if os.path.isdir(self.save_dir + '.zip') or os.path.isfile(self.save_dir + '.zip'):
        #     shutil.rmtree(self.save_dir)
        # else:
        #     print('{} not found. Directory not removed.'.format(self.save_dir + '.zip'))
