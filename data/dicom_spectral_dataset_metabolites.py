import os.path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
# from random import shuffle

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from models.auxiliary import progressbar
from util.util import mkdir


class DicomSpectralMetabolitesDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.ext = opt.input_ext
        counter = 0
        pixels = []
        # spectraI = []
        # Implementing Matlab code from UCSF
        # Compile loaded, reshaped data in row-wise matrix
        if not self.opt.phase_data_path:
            self.A_paths = sorted(make_dataset(self.root, opt)) # Returns a list of paths of the files in the dataset

            for i in progressbar(range(len(self.A_paths)), "Loading patient data: ", 20):
                with pydicom.filereader.dcmread(self.A_paths[i]) as file:
                    data = np.array(file.pixel_array)

                    temp = data.reshape([1,file.Rows,file.Columns,file.NumberOfFrames])
                    map = np.flip(np.flip(temp,0),1)
                    pixels.append(map)

                counter += 1
            pixels = torch.FloatTensor(np.concatenate(pixels,axis=0))#.tolist()
            # spectraI = torch.FloatTensor(np.concatenate(spectraI,axis=0))#.tolist()
            print('------------ Loaded Data ---------------')
            print('pixels.len = ',len(pixels))
            print('pixels.dim = ',pixels.dim())

            length = len(pixels)
            # print('len(Real) = ', len(Real))
            # specLength = len(Real[1])
            mkdir(os.path.join(self.opt.save_dir,'data'))
            path = os.path.join(self.opt.save_dir, 'data/NAA_metabolite_values')
            size = pixels.shape
            metmap = np.empty([size[0], size[1], 1])
            # spectra[:,:,0] = Real
            # spectra[:,:,1] = Imag
            # print('Real and Imaginary components assigned')
            np.save(path,metmap)
            print('Dataset saved as: {}'.format(path))

    def name(self):
        return 'DicomSpectralDataset'
