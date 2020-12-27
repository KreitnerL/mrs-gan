# Basic options that are used to configure either training, validation or testing.
# If an options is not set, its default will be used.

import argparse
import os

import torch
from util import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--split', action='store_true', default=False, help='split the dataset into training, validating, and testing segments')
        self.parser.add_argument('--test_split', type=float, default=0.0, help='percent of data to use for testing') #default=0.1
        self.parser.add_argument('--phase_data_path', type=str, help='if data has already been split, indicate the path for the data split index')#, default='./tests/SpectraGAN_test_012/data')#, default='/home/john/Documents/Research/SpectraGAN/tests/SpectraGAN_test_010/data')
        self.parser.add_argument('--CpoolSize', type=int, default=50, help='critic value pool size for relative loss')
        self.parser.add_argument('--input_dim', type=int, default=1, help='dimension of input data - "1" for spectra, "2" for images,"3" for image volumes')
        self.parser.add_argument('--real', action="store_true", default=False, help='Use only the real portion of the signal')
        self.parser.add_argument('--imag', action="store_true", default=False, help='Use only the real portion of the signal')
        self.parser.add_argument('--mag', action="store_true", default=False, help='Use the magnitude of the spectra')
        self.parser.add_argument('--shuffle_data', action='store_true', help='Shuffle sequence of data when initially extracted from dataset')
        self.parser.add_argument('--normalize', action='store_true', default=False, help='Normalize the input data')
        self.parser.add_argument('--standardize', action='store_true', default=False, help='Standardize the input data')
        self.parser.add_argument('--norm_range', type=list, default=[-1, 1], help='Range in which the input data should be normalized')
        self.parser.add_argument('--pad_data', type=int, default=0, help='Pad data when loading. Most ResNet architectures require padding MRS data by 21')
        self.parser.add_argument('--roi', type=str, default='0,-1', help="Region of interest for spectra")

        self.parser.add_argument('--quiet', action='store_true', default=False, help='Does not print the options in the terminal when initializing')
        self.parser.add_argument('--plot_grads', action='store_true', default=False, help='Plot the gradients for each network after the backward step')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        self.parser.add_argument('--network_stage', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--dataroot', type=str, required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')#
        self.parser.add_argument('--nef', type=int, default=100, help='# of extrator filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='spectra', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=int, default=6, help='number of resnet block for the generator')
        self.parser.add_argument('--which_model_feat', type=str, default='resnet34', help='selects model to use for feature network')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers for the discriminator')
        self.parser.add_argument('--n_layers_E', type=int, default=3, help='number of layers for the extractor')
        self.parser.add_argument('--cbamG', action='store_true', help='Use the convolutional block attention module for the Generator')
        self.parser.add_argument('--cbamD', action='store_true', help='Use the convolutional block attention module for the Discriminator')
        self.parser.add_argument('--n_downsampling', type=int, default=3, help='Number of down-/upsampling steps in the Generator')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='spectra_component_dataset', help='chooses how datasets are loaded.  [dicom_spectral_dataset, spectra_component_dataset]')
        self.parser.add_argument('--model', type=str, default='cycleGAN_W_REG', help='chooses which model to use. [cycleGAN, cycleGAN_W, cycleGAN_W_REG]')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='model checkpoints are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--shuffle', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', default=False, help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--ppm_range', type=str, default='7.171825,-0.501875', help='ppm range for the spectra')

        self.initialized = True

    def get_defaults(self):
        """
        Returns all default values for the options
        """
        args = self.parser.parse_args()
        all_defaults = dict()
        for key in vars(args):
            all_defaults[key] = self.parser.get_default(key)
        return argparse.Namespace(**all_defaults)

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        # save to the disk
        if self.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        if not self.opt.quiet:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        self.opt.ppm_range = list(map(float, self.opt.ppm_range.split(',')))
        self.opt.roi = slice(*list(map(int, self.opt.roi.split(','))))
        
        torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
