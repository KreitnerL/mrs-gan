import argparse
import os
import pickle
import shutil
from collections import OrderedDict

import matplotlib.pyplot as plt
import scipy.io as scp
import util.util as util
from models.auxiliary import accuracy
from models.define import *
from tensorboardX import SummaryWriter  # , add_scalar
from torch.distributions.normal import Normal

# from util.image_pool import ImagePool
from .base_model import BaseModel

__all__ = ['AutoencoderModel']


class AutoencoderModel(BaseModel):
    def name(self):
        return 'Autoencoder'

    def initialize(self, opt, *k):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.error_dict = OrderedDict([('idt', []), ('feat', []), ('loss', [])])
        self.val_error_dict = OrderedDict([('idt', []), ('feat', []), ('loss', [])])

        self.old_lr = opt.lr
        self.device = 'cuda' if len(self.gpu_ids)>0 else 'cpu'


        # Define the Autoencoder
        self.netEn = define.Encoder(dim=self.opt.input_dim, input_nc=opt.input_nc, output_nc=opt.output_nc, n_blocks= opt.n_blocks, ngf=opt.ngf,
                                    norm=opt.AE_norm, actvn=opt.AE_actvn, padding=opt.AE_pad, gpu_ids=opt.gpu_ids, use_dropout=False, se=opt.se)
        self.netDe = define.Decoder(dim=self.opt.input_dim, input_nc=opt.output_nc, output_nc=opt.input_nc, n_blocks= opt.n_blocks, ngf=opt.ngf,
                                    norm=opt.AE_norm, actvn=opt.AE_actvn, padding=opt.AE_pad, gpu_ids=opt.gpu_ids, use_dropout=False, se=opt.se)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netEn, 'En', which_epoch)
            self.load_network(self.netDe, 'De', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_pool = ImagePool(opt.pool_size)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = mse_loss
            self.optimizer = create_optimizer(network=[self.netEn, self.netDe], type='AE', opt=self.opt)


        print('------------ Networks initialized ------------')
        self.num_params = 0
        print_network(self, self.netEn)
        print_network(self, self.netDe)
        print('Total number of Autoencoder network parameters: %d' % self.num_params)

        # Save as text file
        file_name = os.path.join(self.checkpoints_dir, 'model_architecture.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------- Architecture --------------\n')
            opt_file.write('_-------------  Encoder   ---------------\n')
            opt_file.write(print_network(self,self.netEn, file=True))
            opt_file.write('--------------- Decoder -----------------\n')
            opt_file.write(print_network(self,self.netDe, file=True))
            opt_file.write('Total number of Autoencoder network parameters: %d\n' % (self.num_params / 2))
            opt_file.write('----------------- End -------------------\n')

        print('----------------------------------------------')

        self.save('initialized')

    def update_optimizer(self, opt):
        self.optimizer = create_optimizer(network=[self.netEn, self.netDe], type='gen', opt=opt)

    def set_input(self, input): # Dataset_mode Single, non-Single
        # self.data = input.float().to(self.device) if self.opt.k_folds else input['A'].float().to(self.device)
        # input = np.asarray(input)#.astype('float')
        # self.data = torch.from_numpy(input).to(self.device) if self.opt.k_folds else
        self.data = input['magnitude'].float().to(self.device)

        # self.data = input['A'].float().to(self.device)
    def forward(self):
        self.requires_grad(self.netEn, flag=True)
        self.requires_grad(self.netDe, flag=True)
        self.encoded = self.netEn.forward(self.data)    # Noise vector
        self.decoded = self.netDe.forward(self.encoded)

    def test(self):
        self.encoded = self.netEn.forward(self.data).detach()    # Noise vector
        self.decoded = self.netDe.forward(self.encoded).detach()

    def backward_AE(self):
        self.loss_AE()
        self.loss_idt.backward(retain_graph=True)
        self.loss_feat.backward(retain_graph=True)

        self.loss = self.loss_idt * self.opt.lambda_idt + self.loss_feat * self.opt.lambda_mse
        self.loss.backward()#retain_graph=True)
        if self.opt.plot_grads:
            plot_grad_flow(self.netEn, self.debugdir)
            plot_grad_flow(self.netDe, self.debugdir)

    def loss_AE(self):
        assert(self.opt.idt_loss==True or self.opt.mse_loss==True)
        self.loss_idt  = self.criterionIdt(self.decoded,self.data) if self.opt.idt_loss==True else 0
        self.loss_feat = self.criterionFeat(self.decoded,self.data) if self.opt.mse_loss==True else 0
        self.loss = self.loss_idt * self.opt.lambda_idt + self.loss_feat * self.opt.lambda_mse

    def optimize_parameters(self): #
        if len(self.gpu_ids)>0:
            assert(torch.cuda.is_available())

        # Forward
        self.forward()

        self.optimizer.zero_grad()
        self.backward_AE()
        self.optimizer.step()

    def plot_gradients(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_AE()
        plot_grad_flow(self.netDe, self.debugdir, epoch)

    def validate(self):
        self.test()
        self.loss_AE()

    def get_current_errors(self):
        if self.opt.idt_loss==True and self.opt.mse_loss==True:
            idt = self.loss_idt#.data
            feat = self.loss_feat.data
            loss = self.loss.data
            return OrderedDict([('idt', [idt]), ('feat', [feat]), ('loss', [loss])])
        elif self.opt.idt_loss==False:
            feat = self.loss_feat.data
            return OrderedDict([('feat', [feat])])
        elif self.opt.mse_loss==False:
            idt = self.loss_idt#.data
            return OrderedDict([('idt', [idt])])

    def get_current_visuals(self, *opt): # Cycle, non-cycle: identity, non-identity
        if not opt:
            real = util.tensor2plot(self.data)#.data)
            fake = util.tensor2plot(self.noise.data)
            noise = util.tensor2plot(self.input_A)

            return OrderedDict([('real_data', real), ('fake_data', fake)])
        elif opt:
            real = util.tensor2plot(self.data)#.data)
            fake = util.tensor2plot(self.noise.data)
            noise = util.tensor2plot(self.input_A)

            return OrderedDict([('real_data', real), ('fake_data', fake)])

    def save(self, label): # Cycle, non-cycle: encoder, non-encoder
        self.save_network(self.netEn, 'En', label, self.opt.gpu_ids)
        self.save_network(self.netDe, 'De', label, self.opt.gpu_ids)

    def save_train(self, errors, epoch, *batch):
        self.label = 'training'
        self.save_errors(errors, epoch, self.label, dir=self.lossdir)
        for key, value in errors.items():
            self.error_dict[key].append(value.copy())
        self.plotloss(self.error_dict,os.path.join(self.lossdir,'dynamic_training_loss_plots'))

    def save_spectra(self, epoch, batch, label):
        if label == 'training':
            if batch!=None:
                path = os.path.join(self.trainimdir,'epoch_{}_batch_{}'.format(epoch, batch))
            else:
                path = os.path.join(self.trainimdir,'epoch_{}'.format(epoch))
        elif label == 'validation':
            path = os.path.join(self.valimdir,'epoch_{}_batch_{}'.format(epoch, batch))
        # print('save_spectra.path: ',path)
        self.plotspectra(original=self.data,generated=self.decoded,savepth=path, phase=label)

    def save_val(self, errors, fold):
        self.label = 'validation'
        # self.save_errors(errors, fold, self.label, dir=self.vallossdir)
        for key, value in errors.items():
            self.val_error_dict[key].append(value.copy())
        self.plotloss(self.val_error_dict,os.path.join(self.vallossdir,'dynamic_validation_loss_plots'))
        # for key, value in errors.items():
        #     errors[key] = value.mean()
        self.save_errors(self.val_error_dict, fold, self.label, dir=self.vallossdir)

    def update_learning_rate(self): # Cycle, non-cycle: TTUR, no_TTUR
        lr_decay = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lr_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('Update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def writelog(self, phase, epoch):   #TODO: takes multiple lr inputs - convert to variable number
        self.wlog(phase, epoch, [self.opt.lr])

    def plotloss(self, errors, savedir):
        if self.opt.idt_loss==True:
            idt = np.squeeze(np.asarray(torch.FloatTensor(errors['idt']))).copy()   #.cpu()
            if idt.ndim == 0:
                idt = np.max(np.expand_dims(idt,axis=0))
            plt.figure()
            plt.plot(range(len(errors['idt'])), idt, color='xkcd:royal blue', label='Identity Loss')#, title='GAN Training Loss')
            plt.suptitle('Autoencoder Training Identity Loss')#, xlabel='PPM', ylabel='Intensity')
            plt.savefig(savedir+'_identity_loss.svg',format='svg')
            plt.savefig(savedir+'_identity_loss.png',format='png')

        if self.opt.mse_loss==True:
            feat = np.squeeze(np.asarray(torch.FloatTensor(errors['feat']))).copy()   #.cpu()
            if feat.ndim== 0:
                feat = np.max(np.expand_dims(feat,axis=0))
            plt.figure()
            plt.plot(range(len(errors['feat'])), feat, color='tab:cyan', label='MSE Loss')
            plt.suptitle('Autoencoder Training MSE Loss')#, xlabel='PPM', ylabel='Intensity')
            plt.savefig(savedir+'_feature_loss.svg',format='svg')
            plt.savefig(savedir+'_feature_loss.png',format='png')

        if self.opt.idt_loss==True and self.opt.mse_loss==True:
            # # Unscaled plots
            f, ax = plt.subplots(1,1)#, constrained_layout=True)#, sharex='all', sharey='all')#1)
            ax.set_aspect('auto')
            ax.plot(range(len(errors['idt'])), idt, color='xkcd:royal blue', label='Identity Loss')#, title='GAN Training Loss')
            ax.plot(range(len(errors['feat'])), feat, color='tab:cyan', label='MSE Loss')
            ax.set(title='Autoencoder Training Losses')#, xlabel='PPM', ylabel='Intensity')
            ax.legend(loc=1)

            f.tight_layout(h_pad=1)
            plt.savefig(savedir+'.svg',format='svg', bbox_inches='tight')
            plt.savefig(savedir+'.png',format='png', bbox_inches='tight')

            path = savedir + '_combined_loss_axes.pkl'
            file = open(path,'wb')
            pickle.dump(ax, file, protocol=4)
            file.close()

            plt.figure()
            plt.plot(range(len(errors['loss'])), np.squeeze(np.asarray(errors['loss'])), color='xkcd:royal blue', label='Combined Loss')
            plt.suptitle('Autoencoder Training Combined Loss')#, xlabel='PPM', ylabel='Intensity')
            plt.savefig(savedir+'_combined_loss.svg',format='svg')
            plt.savefig(savedir+'_combined_loss.png',format='png')

            # # Scaled plots
            f, ax = plt.subplots(1,1)#, constrained_layout=True)#, sharex='all', sharey='all')#1)
            ax.set_aspect('auto')
            idt = (idt - np.min(idt)) / (np.max(idt) - np.min(idt))
            feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
            ax.plot(range(len(errors['idt'])), idt, color='xkcd:royal blue', label='Identity Loss')#, title='GAN Training Loss')
            ax.plot(range(len(errors['feat'])), feat, color='tab:cyan', label='MSE Loss')
            ax.set(title='Autoencoder Training Losses - Scaled')#, xlabel='PPM', ylabel='Intensity')
            ax.legend(loc=1)

            f.tight_layout(h_pad=1)
            plt.savefig(savedir+'_scaled.svg',format='svg', bbox_inches='tight')
            plt.savefig(savedir+'_scaled.png',format='png', bbox_inches='tight')

            path = savedir + '_combined_loss_axes_scaled.pkl'
            file = open(path,'wb')
            pickle.dump(ax, file, protocol=4)
            file.close()

        plt.close('all')
        del idt, feat

    def plotspectra(self, original, generated, savepth, phase, individual=None):
        lw = 0.5
        ratio = 1

        if phase=='training':
            index = [0]
        elif phase=='validation':
            index = torch.randint(low=0,high=original.size(dim=0),size=[2]).tolist()
        else:
            index = [range(original.size(dim=0))]

        original, generated = np.asarray(original.cpu()), np.asarray(generated.detach().cpu())

        for _, i in enumerate(index):
            if individual==False or individual==None:
                # # Combined original and generated spectra
                f = plt.figure()
                f.suptitle('Training Profile of Spectra - Real\n')
                ax = f.add_subplot(111)
                ax.set_aspect(1/ax.get_data_ratio()*ratio)
                ax.plot(generated[i,0,:], color='r', linewidth=lw, label='Reconstructed')
                ax.plot(original[i,0,:], color='xkcd:dark blue', linewidth=lw, label='Original')
                ax.set(xlabel='PPM', ylabel='Intensity')
                ax.legend(loc=1)
                # f.tight_layout(h_pad=1)#rect=[0, 0.3, 1, 0.95])
                base = '_comb_real' if i==0 else '_comb_real_{}'.format(i)
                path = savepth + base + '_axes.pkl'
                with open(path, 'wb') as file:
                    pickle.dump(ax, file, protocol=4)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg', bbox_inches='tight')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png', bbox_inches='tight')

                f = plt.figure()
                ax = f.add_subplot(111)
                ax.set_aspect(1/ax.get_data_ratio()*ratio)
                f.suptitle('Training Profile of Spectra - Imaginary\n')
                ax.plot(generated[i,1,:], color='r', linewidth=lw, label='Reconstructed')
                ax.plot(original[i,1,:], color='xkcd:dark blue', linewidth=lw, label='Original')
                ax.set(xlabel='PPM', ylabel='Intensity')
                ax.legend(loc=1)
                # f.tight_layout(h_pad=1)
                base = '_comb_imag' if i==0 else '_comb_imag_{}'.format(i)
                path = savepth + base + '_axes.pkl'
                with open(path, 'wb') as file:
                    pickle.dump(ax, file, protocol=4)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg', bbox_inches='tight')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png', bbox_inches='tight')


            if individual==True or individual==None:
                f = plt.figure()
                f.suptitle('Training Profile of Spectra - Real\n')
                ax = f.add_subplot(111)
                ax.plot(original[i,0,:], color='xkcd:dark blue', linewidth=lw)
                ax.set(title='\nOriginal Spectra', xlabel='PPM', ylabel='Intensity')
                f.tight_layout(h_pad=1)
                base = '_original_real' if len(index)==0 else '_original_real_{}'.format(i)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png')
                path = savepth + base + '_axes.pkl'
                file = open(path,'wb')
                pickle.dump(ax, file, protocol=4)
                file.close()

                f = plt.figure()
                f.suptitle('Training Profile of Spectra - Imaginary\n')
                ax = f.add_subplot(111)
                ax.plot(original[i,1,:], color='xkcd:dark blue', linewidth=lw)
                ax.set(title='\nOriginal Spectra', xlabel='PPM', ylabel='Intensity')
                f.tight_layout(h_pad=1)
                base = '_original_imag' if len(index)==0 else '_original_imag_{}'.format(i)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png')
                path = savepth + base + '_axes.pkl'
                file = open(path,'wb')
                pickle.dump(ax, file, protocol=4)
                file.close()

                f = plt.figure()
                f.suptitle('Training Profile of Spectra - Real\n')
                ax = f.add_subplot(111)
                ax.plot(generated[i,0,:], color='xkcd:dark blue', linewidth=lw)
                ax.set(title='\nGenerated Spectra', xlabel='PPM', ylabel='Intensity')
                f.tight_layout(h_pad=1)
                base = '_generated_real' if len(index)==0 else '_generated_real_{}'.format(i)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png')

                path = savepth + base + '_axes.pkl'
                file = open(path,'wb')
                pickle.dump(ax, file, protocol=4)
                file.close()

                f = plt.figure()
                f.suptitle('Training Profile of Spectra - Imaginary\n')
                ax = f.add_subplot(111)
                ax.plot(generated[i,1,:], color='xkcd:dark blue', linewidth=lw)
                ax.set(title='\nGenerated Spectra', xlabel='PPM', ylabel='Intensity')
                f.tight_layout(h_pad=1)
                base = '_generated_imag' if len(index)==0 else '_generated_imag_{}'.format(i)
                path = savepth + base + '_fig.svg'
                plt.savefig(path, format='svg')
                path = savepth + base + '_fig.png'
                plt.savefig(path, format='png')

                path = savepth + base + '_axes.pkl'
                file = open(path,'wb')
                pickle.dump(ax, file, protocol=4)
                file.close()
            plt.close('all')

    def wlog(self, phase, epoch, lr):
        assert(len(lr)==1)
        lr = lr[0]
        while self.log==0:
            self.writer = SummaryWriter(os.path.join(self.opt.save_dir,'log'))
            self.log = 1
        if self.log==1:
            if phase=='train':
                save_filename = 'Epoch_%s_%s_errors.pkl' % (epoch, 'training')
                save_path = os.path.join(self.save_dir, save_filename)
                if os.path.isfile(save_path):
                    with open(save_path, 'rb') as file:
                        unpickler = pickle.Unpickler(file)
                        dict = unpickler.load()
                        for i in range(len(dict['loss'])):
                            self.writer.add_scalar('Train/loss', dict['loss'][i], i)
                            self.writer.add_scalar('Train/idt_loss', dict['idt'][i], i)
                            self.writer.add_scalar('Train/feat_loss', dict['feat'][i], i)
                        self.writer.add_scalar('Train/learning_rate', lr, epoch)

            elif phase=='val':
                save_filename = 'Fold_{}_validation_errors.pkl'.format(epoch)
                save_path = os.path.join(self.save_dir, save_filename)
                if os.path.isfile(save_path):
                    with open(save_path, 'rb') as file:
                        unpickler = pickle.Unpickler(file)
                        dict = unpickler.load()
                        for i in range(len(dict['D'])):
                            self.writer.add_scalar('Validation/loss', dict['loss'][i], i)
                            self.writer.add_scalar('Validation/idt_loss', dict['idt'][i], i)
                            self.writer.add_scalar('Validation/feat_loss', dict['feat'][i], i)
                        self.writer.add_scalar('Validation/learning_rate', lr, epoch)
                self.writer.close()
        if epoch==(self.opt.niter + self.opt.niter_decay + 1):
            self.writer.close()

    def save_errors(self, errors, epoch, label, dir):
        # save_filename = '%s_epoch_%s_%s_errors.pkl' % (name, epoch, label)
        save_filename = label + '_errors.pkl'
        save_path = os.path.join(dir, save_filename)
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as file:
                unpickler = pickle.Unpickler(file)
                dict = unpickler.load()
            with open(save_path, 'wb') as file:
                for key, value in errors.items():
                    dict[key].append(value)
                pickle.dump(dict, file, pickle.HIGHEST_PROTOCOL)
            scp.savemat(os.path.join(dir,label + '_errors.mat'), mdict=dict)
        else:
            with open(save_path, 'wb') as file:
               pickle.dump(errors, file, pickle.HIGHEST_PROTOCOL)
            scp.savemat(os.path.join(dir,label + '_errors.mat'), mdict=errors)
