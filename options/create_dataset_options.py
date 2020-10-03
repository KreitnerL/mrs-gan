# Basic options that are used to configure either training, validation or testing.
# If an options is not set, its default will be used.

import argparse
import os

import torch
from util import util

class CreateDatasetOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--source_dir_A', type=str, help='Directory of the dataset of domain A')
        self.parser.add_argument('--source_dir_B', type=str, help='Directory of the dataset of domain B')
        self.parser.add_argument('--save_dir', type=str, help='Directory where the dataset will be saved')
        self.parser.add_argument('--file_ext_spectra', type=str, default='proc.dcm', help='File extension of the processed spectra DICOM files')
        self.parser.add_argument('--file_ext_metabolic_map', type=str, default='NAA.dcm', help='File extension of the metabolic map')
        self.parser.add_argument('--force', type=bool, default=False, help='If true, overwrites all exisiting .npz, .mat, .dat files')

        self.parser.add_argument('--normalize', type=bool, default=False, help='normalize the spectra in preprocessing')
        self.parser.add_argument('--standardize', type=bool, default=False, help='standardize the spectra in preprocessing')
        self.parser.add_argument('--pad_data', type=bool, default=False, help='pad_data the spectra in preprocessing')
        self.parser.add_argument('--input_nc', type=int, default=2, help='number if input channels')
        self.parser.add_argument('--real', type=bool, default=False, help='only use real part of the spectra')
        self.parser.add_argument('--imag', type=bool, default=False, help='only use imaginary part of the spectra')
        self.parser.add_argument('--split', type=bool, default=True, help='Split the data into train, validation and test set')
        self.parser.add_argument('--val_split', type=float, default=0.3, help='Part of dataset that is used for validation')
        self.parser.add_argument('--test_split', type=float, default=0, help='Part of dataset that is used for testing') 
        self.parser.add_argument('--shuffle_data', type=bool, default=True, help='Select spectra for training / testing randomly')
        self.parser.add_argument('--quiet', action='store_true', default=False, help='Does not print the options in the terminal when initializing')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        if not self.opt.quiet:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        return self.opt
