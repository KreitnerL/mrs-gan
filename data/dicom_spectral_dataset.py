import functools
import os
import os.path
import numpy as np
from torch import from_numpy

from data.base_dataset import BaseDataset

index = {'train': 0, 'val': 1, 'test': 2}

class DicomSpectralDataset(BaseDataset):
    """
    DicomSpectralDataset loads spectra from .dat files and returns a sample as a numpy array
    """
    def initialize(self, opt):
        self.opt = opt
        self.roi = slice(self.opt.crop_start,self.opt.crop_end)
        self.root = opt.dataroot
        if opt.real:
            self.channel_index = slice(0,1)
        elif opt.imag:
            self.channel_index = slice(1,2)
        else:
            self.channel_index = slice(None, None)

        if self.opt.phase == 'val':
            return self.init_val(opt)
        if self.opt.phase == 'test':
            self.opt.phase = 'val'

        sizes_A = np.genfromtxt(os.path.join(self.root,'sizes_A') ,delimiter=',').astype(np.int64)
        sizes_B = np.genfromtxt(os.path.join(self.root,'sizes_B') ,delimiter=',').astype(np.int64)

        path_A = str(os.path.join(self.root, self.opt.phase + '_A.dat'))
        path_B = str(os.path.join(self.root, self.opt.phase + '_B.dat'))

        self.A_size = sizes_A[index[self.opt.phase]]
        self.B_size = sizes_B[index[self.opt.phase]]
        self.length = sizes_A[3]

        self.sampler_A = np.memmap(path_A, dtype='double', mode='r', shape=(self.A_size,sizes_A[4],sizes_A[3]))
        self.sampler_B = np.memmap(path_B, dtype='double', mode='r', shape=(self.B_size,sizes_B[4],sizes_B[3]))
        if self.opt.phase == 'val':
            self.opt.phase = 'test'

        self.innit_transformations()

    def init_val(self, opt):
        self.letter = 'A' if opt.AtoB else 'B'
        self.dir = os.path.join(opt.dataroot, opt.phase + self.letter)
        sizes = np.genfromtxt(os.path.join(self.root,'sizes_' + self.letter) ,delimiter=',').astype(np.int64)
        self.length = sizes[3]
        path = str(os.path.join(self.root, self.opt.phase + '_{0}.dat'.format(self.letter)))
        self.size = sizes[index[self.opt.phase]]
        self.sampler = np.memmap(path, dtype='double', mode='r', shape=(self.size,sizes[4],sizes[3]))
        self.innit_transformations()

    def innit_transformations(self):
        self.transformations = [lambda A: np.asarray(A).astype(float)]
        if self.opt.mag:
            self.transformations.append(lambda A: np.expand_dims(np.sqrt(A[0,:]**2 + A[1,:]**2), 0))
        self.transformations.append(lambda A: A/np.amax(abs(A)))
        self.transformations.append(from_numpy)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        if self.opt.phase != 'val':
            A = self.sampler_A[index % self.A_size,self.roi]
            B = self.sampler_B[index % self.B_size,self.roi]
            return {
                'A': self.transform(A),
                'B': self.transform(B),
                'A_paths': '{:03d}.foo'.format(index % self.A_size),
                'B_paths': '{:03d}.foo'.format(index % self.B_size)
            }
        else:
            data = self.sampler[index % self.size,self.channel_index,self.roi]
            return {
                self.letter: self.transform(data),
                'A_paths': '{:03d}.foo'.format(index % self.size)
            }

    def __len__(self):
        if self.opt.phase != 'val':
            return max(self.A_size, self.B_size) # Determines the length of the dataloader
        else:
            return self.size
    
    def get_length(self):
        if self.opt.crop_start != None and self.opt.crop_end != None:
            l = self.opt.crop_end - self.opt.crop_start
        else:
            l = self.length
        # length must be power of 2
        assert (l & (l-1) == 0) and l != 0
        return l

    def name(self):
        return 'DicomSpectralDataset'

    def transform(self, data):
        return functools.reduce((lambda x, y: y(x)), self.transformations, data)