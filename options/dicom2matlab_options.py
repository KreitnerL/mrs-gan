# Basic options that are used to configure either training, validation or testing.
# If an options is not set, its default will be used.

import argparse

class Dicom2MatlabOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--source_dir', type=str, help='Directory of the dicom dataset')
        self.parser.add_argument('--save_dir', type=str, help='Directory where the matlab dataset will be saved')
        self.parser.add_argument('--file_ext_spectra', type=str, default='proc.dcm', help='File extension of the processed spectra DICOM files')
        self.parser.add_argument('--file_ext_metabolic_map', type=str, default='NAA.dcm', help='File extension of the metabolic map')
        self.parser.add_argument('--force', action='store_true', help='If true, overwrites all exisiting .npz, .mat, .dat files')

        self.parser.add_argument('--normalize', type=bool, default=False, help='normalize the spectra in preprocessing')
        self.parser.add_argument('--standardize', type=bool, default=False, help='standardize the spectra in preprocessing')
        self.parser.add_argument('--pad_data', type=bool, default=False, help='pad_data the spectra in preprocessing')
        self.parser.add_argument('--real', action='store_true', help='only use real part of the spectra')
        self.parser.add_argument('--imag', action='store_true', help='only use imaginary part of the spectra')
        self.parser.add_argument('--mag', action='store_true', help='only use the magnitude of the spectra')
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