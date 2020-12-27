# Basic options that are used to configure either training, validation or testing.
# If an options is not set, its default will be used.

import argparse

class CreateDatasetOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--source_path_A', type=str, help='File path of the dataset of domain A')
        self.parser.add_argument('--source_path_B', type=str, help='File path of the dataset of domain B')
        self.parser.add_argument('--save_dir', type=str, help='Directory where the dataset will be saved')
        self.parser.add_argument('--source_path_source_labels', type=str, help='File path of the labels of the target domain')
        self.parser.add_argument('--A_mat_var_name', type=str, default='spectra', help='Name of the matlab variable containing the spectra for domain A')
        self.parser.add_argument('--B_mat_var_name', type=str, default='spectra', help='Name of the matlab variable containing the spectra for domain B')
        self.parser.add_argument('--label_names', type=str, default='cho,naa', help='Name of the dataset')

        self.parser.add_argument('--split', type=bool, default=True, help='Split the data into train, validation and test set')
        self.parser.add_argument('--val_split', type=float, default=0.1, help='Part of dataset that is used for validation')
        self.parser.add_argument('--test_split', type=float, default=0, help='Part of dataset that is used for testing') 
        self.parser.add_argument('--shuffle_data', action='store_true', help='Select spectra for training / testing randomly')
        self.parser.add_argument('--quiet', action='store_true', default=False, help='Does not print the options in the terminal when initializing')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        self.opt.label_names = self.opt.label_names.split(',')
        self.opt.train_indices, self.opt.val_indices, self.opt.test_indices = None, None, None

        if not self.opt.quiet:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
        return self.opt
