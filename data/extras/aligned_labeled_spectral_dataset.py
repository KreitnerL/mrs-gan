import os.path

from pandas import read_csv

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class AlignedLabeledSpectralDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))


    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        with open(AB_path,"r") as file:
            AB_spectra = read_csv(file, names=['Spectra','NAA','cho'])

        A = AB_spectra.Spectra
        B = [AB_spectra.NAA, AB_spectra.cho]

        return {'A_data': A, 'B_labels': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedLabeledSpectralDataset'
