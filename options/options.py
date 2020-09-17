import argparse
import os

import torch
from util import util


__all__ = ['BaseOptions', 'TrainOptions', 'TestOptions']


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/home/john/Documents/Research/SpectraGAN/datasets/')#/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#default='/home/john/SpectraGAN/tests/SpectraGAN_test_012/data')#default='/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#DatasetFile/train')#required=True, # '/home/john/SpectraGAN/datasets/spectra')#
        self.parser.add_argument('--split', action='store_true', default=True, help='split the dataset into training, validating, and testing segments')
        self.parser.add_argument('--val_split', type=float, default=0.2, help='percent of data to use for validation')
        self.parser.add_argument('--test_split', type=float, default=0.1, help='percent of data to use for testing') #default=0.1
        self.parser.add_argument('--phase_data_path', type=str, help='if data has already been split, indicate the path for the data split index')#, default='./tests/SpectraGAN_test_012/data')#, default='/home/john/Documents/Research/SpectraGAN/tests/SpectraGAN_test_010/data')
        self.parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
        self.parser.add_argument('--CpoolSize', type=int, default=50, help='critic value pool size for relative loss')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
        self.parser.add_argument('--input_dim', type=int, default=1, help='dimension of input data - "1" for spectra, "2" for images,"3" for image volumes')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netEst', type=str, default='resnet', help='selects model to use for netEst: resnet or dense')
        self.parser.add_argument('--which_model_netEn', type=str, default='conv_encoder')
        self.parser.add_argument('--flexible', action='store_true', default=False, help='Use flexible, learned convolutions and linear layers')
        self.parser.add_argument('--pretrained_G', action='store_true', help='use a pretrained generator model')
        self.parser.add_argument('--n_layers_g', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--downsampling', type=int, default=2, help='Downsampling factor for the generator network')
        self.parser.add_argument('--real', action="store_false", default=True, help='Use only the real portion of the signal')
        self.parser.add_argument('--imag', action="store_false", default=True, help='Use only the real portion of the signal')
        self.parser.add_argument('--G0', action='append_const', dest='gpu_ids', const=0, help='use GPU 0 in addition to the default GPU')
        self.parser.add_argument('--G1', action='append_const', dest='gpu_ids', const=1, help='use GPU 1 in addition to the default GPU')
        self.parser.add_argument('--G2', action='append_const', dest='gpu_ids', const=2, help='use GPU 2 in addition to the default GPU')
        self.parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')#'experiment_name'
        self.parser.add_argument('--dataset_mode', type=str, default='SpectralDataset', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='fitting', help='Chooses which model to use.')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/checkpoints', help='models are saved here')

        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # self.parser.add_argument('--logdir', type=str, default='/log', help='models are saved here')
        self.parser.add_argument('--cropped_signal', action='store_true', default=False, help='crop spectra to Long Echo region: 1024 - [500:625]; 2048 - [1000:1256]')
        self.parser.add_argument('--cropRange', type=tuple, default=(1005,1325))
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--actvn', type=str, default='relu', help='activation method: relu, tanh, selu, prelu, rrelu, none')
        self.parser.add_argument('--pad', type=str, default='reflect', help='padding method: reflection, replication, zero')

        self.parser.add_argument('--use_sigmoid', action='store_true', default=False, help='use sigmoid or not')# Todo: finish this option
        self.parser.add_argument('--serial_batches', action='store_true', default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--shuffle_data', action='store_true', default=True, help='Shuffle sequence of data when initially extracted from dataset')
        self.parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default=None, help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--transform', action='store_true', default=False, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--use_dropout', action='store_true', help='no dropout for the generator')

        # # Extras
        self.parser.add_argument('--quiet', action='store_true', default=False, help='Does not print the options in the terminal when initializing')
        self.parser.add_argument('--se', action='store_true', default=False, help='Use squeeze-and-excitation blocks')
        self.parser.add_argument('--pAct', action='store_true', default=False, help='Pre-activate the residual units')
        self.parser.add_argument('--depth', type=int, default=1, help='# of residual blocks per layer')
        self.parser.add_argument('--learned_g', action='store_true', default=False, help='learn the convolutional grouping strategy')

        self.parser.add_argument('--plot_grads', action='store_true', default=False, help='Plot the gradients for each network after the backward step')
        self.parser.add_argument('--normalize', action='store_true', default=False, help='Normalize the input data')
        self.parser.add_argument('--standardize', action='store_true', default=False, help='Standardize the input data')
        self.parser.add_argument('--norm_range', type=list, default=[-1, 1], help='Range in which the input data should be normalized')

        self.parser.add_argument('--pad_data', type=int, default=0, help='Pad data when loading. Most ResNet architectures require padding MRS data by 21')
        # # Multi-Scale
        self.parser.add_argument('--num_down', type=int, default=5, help='Number of layers to downsample and upsample the architecture')
        self.parser.add_argument('--n_blocks', type=int, default=6, help='Number of residual units in each dense block')
        self.parser.add_argument('--growth_rate', type=int, default=3, help='Number of dense connections within each dense block')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.opt.gpu_ids:
            length = len(self.opt.gpu_ids)
            if length == 1:
                self.opt.gpu_ids = [int(self.opt.gpu_ids[0])]
            elif length == 2:
                self.opt.gpu_ids = [int(self.opt.gpu_ids[0]), int(self.opt.gpu_ids[1])]
            elif length == 3:
                self.opt.gpu_ids = [int(self.opt.gpu_ids[0]), int(self.opt.gpu_ids[1]), int(self.opt.gpu_ids[2])]
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            self.opt.gpu_ids = []


        args = vars(self.opt)

        if not self.opt.quiet:
            print('--------------- Options ----------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('----------------- End ------------------')

        # save to the disk
        self.opt.save_dir = os.path.join('./tests/',self.opt.name)
        self.opt.checkpoints_dir = os.path.join(self.opt.save_dir, 'checkpoints')
        util.mkdirs(self.opt.checkpoints_dir)
        file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

    def bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--file_ext', type=str, default='proc.npz', help='can add additional information to select specific files from the dataset')
        self.parser.add_argument('--input_ext', type=str, default='proc.npz', help='can add additional information to select specific files from the dataset')
        self.parser.add_argument('--folder_ext', type=str, default='UCSF', help='can add additional information to select specific folders from the dataset')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=25000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--starting_epoch', type=int, default=1, help='when training, or continuing training, which epoch should start first?')
        self.parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=20, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.900, help='momentum term of adam for generator')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam for discriminator')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')#required='no_TTUR',
        self.parser.add_argument('--lr_method', type=str, default=None, help='set the learning rate scheduler: None, cosine, etc.')
        # self.parser.add_argument('--warmup', type=int, default=None, help='number of epochs to use to warm up the learning rate')
        self.parser.add_argument('--warmup', action='store_true', default=False, help='number of epochs to use to warm up the learning rate')
        # # Parameter Loss
        self.parser.add_argument('--parameters', action='store_true', default=False,help='Calculate loss based on parameters')
        self.parser.add_argument('--metabolites', action='store_true', default=False, help='Add a loss emphasis to the 3 metabolite lines')
        self.parser.add_argument('--lambda_metab', type=int, default=0, help='weight for the metabolite quantities')
        # # Spectrum Loss
        self.parser.add_argument('--spectrum', action='store_true',default=False,help='Calculate loss based on spectra')
        # # Identity Loss
        self.parser.add_argument('--idt', action='store_true', default=False, help='Calculate an identity loss for the generator')
        self.parser.add_argument('--lambda_idt', type=float, default=2, help='weight for the identity loss in the generator')
        self.parser.add_argument('--mse_loss', action='store_true', help='Use MSE loss to train the autoencoder')
        self.parser.add_argument('--lambda_mse', type=float, default=1, help='Lambda coefficient for MSE loss')
        self.parser.add_argument('--phase_loss', action='store_true', default=False, help='Add an additional penalty for phase error')
        self.parser.add_argument('--lambda_phase', type=float, default=1, help='Lambda coefficient for phase loss')
        # # Perceptual Loss
        self.parser.add_argument('--perception', action='store_true', default=False, help='Use perceptual feature loss when training')
        self.parser.add_argument('--lambda_feat_loss', type=float, default=5.0, help='weight for the perceptual feature loss component')
        # # Relative & Profile Entropy Loss
        self.parser.add_argument('--entropy', action='store_true', default=False, help='Calculate an entropy loss for the generator')
        self.parser.add_argument('--h', type=str, default='sel', help='"sel" for squared entropy loss or "el" for general entropy loss')
        self.parser.add_argument('--lambda_h', type=float, default=5, help='weight for the entropy profile loss (KL Divergence)')
        self.parser.add_argument('--runningH', action='store_true', default=False, help='Calculate the entropy across the spectra vs of the entire spectra')
        # self.parser.add_argument('--rHkernel', type=float, default=32, help='Size of the kernel for calculating the running entropy')
        # self.parser.add_argument('--rh', type=str, default='rsel', help='"rsel" for relative squared entropy loss or "rel" for relative entropy loss')
        # self.parser.add_argument('--lambda_rh', type=float, default=5, help='weight for the relative entropy loss')

        self.parser.add_argument('--k_folds', type=int, default=-1, help='number of folds for a cross-validation training scheme')

        self.parser.add_argument('--magnitude', action='store_true', default=False, help='Use magnitude spectra instead of their real component')

        self.isTrain = True



class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.isTrain = False
