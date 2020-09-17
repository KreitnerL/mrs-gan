import os.path

from pandas import read_csv

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class AlignedSpectraSpectralDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))


    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        with open(AB_path,"r") as file:
            AB_spectra = read_csv(file, names=['spectra1','spectra2'])

        A = AB_spectra.spectra1
        B = AB_spectra.spectra2

        return {'A_spectra1': A, 'B_spectra2': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedSpectraSpectralDataset'
