# Basic options that are used to configure either training, validation or testing.
# If an options is not set, its default will be used.

import argparse
import os

import torch
# import util
from util import util


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataroot', type=str, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/home/john/Documents/Research/SpectraGAN/datasets/')#/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#default='/home/john/SpectraGAN/tests/SpectraGAN_test_012/data')#default='/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#DatasetFile/train')#required=True, # '/home/john/SpectraGAN/datasets/spectra')#
# parser.add_argument('--split', action='store_true', default=False, help='split the dataset into training, validating, and testing segments')
# parser.add_argument('--val_split', type=float, default=0.2, help='percent of data to use for validation')
# parser.add_argument('--test_split', type=float, default=0.0, help='percent of data to use for testing') #default=0.1
# parser.add_argument('--phase_data_path', type=str, help='if data has already been split, indicate the path for the data split index')#, default='./tests/SpectraGAN_test_012/data')#, default='/home/john/Documents/Research/SpectraGAN/tests/SpectraGAN_test_010/data')
# parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# args = parser.parse_args()
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='/home/john/Documents/Research/SpectraGAN/datasets/')#/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#default='/home/john/SpectraGAN/tests/SpectraGAN_test_012/data')#default='/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1')#DatasetFile/train')#required=True, # '/home/john/SpectraGAN/datasets/spectra')#
        self.parser.add_argument('--split', action='store_true', default=False, help='split the dataset into training, validating, and testing segments')
        self.parser.add_argument('--val_split', type=float, default=0.2, help='percent of data to use for validation')
        self.parser.add_argument('--test_split', type=float, default=0.0, help='percent of data to use for testing') #default=0.1
        self.parser.add_argument('--phase_data_path', type=str, help='if data has already been split, indicate the path for the data split index')#, default='./tests/SpectraGAN_test_012/data')#, default='/home/john/Documents/Research/SpectraGAN/tests/SpectraGAN_test_010/data')
        self.parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
        self.parser.add_argument('--CpoolSize', type=int, default=50, help='critic value pool size for relative loss')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
        self.parser.add_argument('--input_dim', type=int, default=1, help='dimension of input data - "1" for spectra, "2" for images,"3" for image volumes')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic_spectra', help='selects model to use for netD')
        # self.parser.add_argument('--which_model_netG', type=str, default='resnet_6blocks', help='selects model to use for netG')
        # self.parser.add_argument('--which_model_netAE', type=str, default='msgdenseresnet', help='selects model to use for netG')
        # self.parser.add_argument('--which_model_netEn', type=str, default='conv_encoder')
        self.parser.add_argument('--pretrained_G', action='store_true', help='use a pretrained generator model')
        # self.parser.add_argument('--pretrained_G', action='store_true', help='use a pretrained encoder model')
        self.parser.add_argument('--n_layers_g', type=int, default=3, help='only used if which_model_netD==n_layers')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--downsampling', type=int, default=2, help='Downsampling factor for the generator network')
        self.parser.add_argument('--real', action="store_false", default=True, help='Use only the real portion of the signal')
        self.parser.add_argument('--imag', action="store_false", default=True, help='Use only the real portion of the signal')
        # self.parser.add_argument('--gpu_ids', type=int, default=-1, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # self.parser.add_argument('--gpu_ids', nargs='+', dest='gpu_ids', action='store', help='gpu ids: e.g. 0  0,1,2, 0,2. use [] for CPU')#-1 for CPU')#type=int,
        self.parser.add_argument('--G0', action='append_const', dest='gpu_ids', const=0, help='use GPU 0 in addition to the default GPU')
        self.parser.add_argument('--G1', action='append_const', dest='gpu_ids', const=1, help='use GPU 1 in addition to the default GPU')
        self.parser.add_argument('--G2', action='append_const', dest='gpu_ids', const=2, help='use GPU 2 in addition to the default GPU')
        self.parser.add_argument('--name', type=str, default='SpectraGAN_test_013', help='name of the experiment. It decides where to store samples and models')#'experiment_name'
        self.parser.add_argument('--dataset_mode', type=str, default='DicomSpectralDataset', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='spectra',
                                 help='chooses which model to use. cycle_gan, ra_cyclegan, pix2pix, custom, test, autoencoder')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/checkpoints', help='models are saved here')
        # self.parser.add_argument('--logdir', type=str, default='/log', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        # self.parser.add_argument('--gen_norm', type=str, default='batch', help='normalization method for generator: batch, instance, spectral')
        # self.parser.add_argument('--dis_norm', type=str, default='spectral', help='normalization method for discriminator: batch, instance, spectral')
        # self.parser.add_argument('--AE_norm', type=str, default='batch', help='normalization method for generator: batch, instance, spectral')

        self.parser.add_argument('--gen_actvn', type=str, default='relu', help='activation method for generator: relu, tanh, selu, prelu, rrelu, none')
        # self.parser.add_argument('--dis_actvn', type=str, default='leakyrelu', help='activation method for discriminator: relu, leakyrelu, tanh, selu, prelu, rrelu, none')
        # self.parser.add_argument('--AE_actvn', type=str, default='selu', help='activation method for generator: relu, tanh, selu, prelu, rrelu, none')

        self.parser.add_argument('--gen_pad', type=str, default='reflect', help='padding method for generator: reflection, replication, zero')
        # self.parser.add_argument('--dis_pad', type=str, default='reflect', help='padding method for discriminator: reflection, replication, zero')
        # self.parser.add_argument('--AE_pad', type=str, default='reflect', help='padding method for generator: reflection, replication, zero')

        self.parser.add_argument('--use_sigmoid_G', type=bool, default=False, help='use sigmoid or not')# Todo: finish this option
        # self.parser.add_argument('--use_sigmoid_D', type=bool, default=True, help='use sigmoid or not')# Todo: finish this option
        self.parser.add_argument('--serial_batches', action='store_true', default=True, help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--shuffle_data', action='store_true', default=True, help='Shuffle sequence of data when initially extracted from dataset')
        # self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        # self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        # self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        # self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--dropout', action='store_true', default=False, help='Use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default=None, help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--transform', action='store_true', default=False, help='if specified, do not flip the images for data augmentation')
        
        self.parser.add_argument('--quiet', action='store_true', default=False, help='Does not print the options in the terminal when initializing')
        self.parser.add_argument('--se', action='store_true', default=False, help='Use squeeze-and-excitation blocks')
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
                # self.opt.gpu_ids[0], self.opt.gpu_ids[1] = [int(self.opt.gpu_ids[0])], [int(self.opt.gpu_ids[1])]
                self.opt.gpu_ids = [int(self.opt.gpu_ids[0]), int(self.opt.gpu_ids[1])]
            elif length == 3:
                self.opt.gpu_ids = [int(self.opt.gpu_ids[0]), int(self.opt.gpu_ids[1]), int(self.opt.gpu_ids[2])]

            # print('type and value of gpu_ids = ', type(self.opt.gpu_ids), self.opt.gpu_ids, self.opt.gpu_ids[0])
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
