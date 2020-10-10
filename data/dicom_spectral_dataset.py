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
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        if opt.real:
            self.channel_index = 0 
        elif opt.imag:
            self.channel_index = 1
        else:
            self.channel_index = None
        sizes_A = np.genfromtxt(os.path.join(self.root,'sizes_A') ,delimiter=',').astype(np.int64)
        sizes_B = np.genfromtxt(os.path.join(self.root,'sizes_B') ,delimiter=',').astype(np.int64)

        path_A = str(os.path.join(self.root, self.opt.phase + '_A.dat'))
        path_B = str(os.path.join(self.root, self.opt.phase + '_B.dat'))

        self.A_size = sizes_A[index[self.opt.phase]]
        self.B_size = sizes_B[index[self.opt.phase]]

        self.sampler_A = np.memmap(path_A, dtype='double', mode='r', shape=(self.A_size,sizes_A[4],sizes_A[3]))
        self.sampler_B = np.memmap(path_B, dtype='double', mode='r', shape=(self.B_size,sizes_B[4],sizes_B[3]))
        self.counter=0
        print('Dataset sampler loaded')

    def __getitem__(self, index):
        # 'Generates one sample of data'
        if self.channel_index is not None:
            A = np.expand_dims(np.asarray(self.sampler_A[index % self.A_size,self.channel_index,:]).astype(float),0)
            B = np.expand_dims(np.asarray(self.sampler_B[index % self.B_size,self.channel_index,:]).astype(float),0)
        else:
            A = np.asarray(self.sampler_A[index % self.A_size,:,:]).astype(float)
            B = np.asarray(self.sampler_B[index % self.B_size,:,:]).astype(float)
        return {
            'A': from_numpy(A),
            'B': from_numpy(B),
        }

    def __len__(self):
        return max(self.A_size, self.B_size) # Determines the length of the dataloader

    def name(self):
        return 'DicomSpectralDataset'

    
    # TODO maybe remove
    def extract(self, index):
        label = ['data/training.dat', 'data/validation.dat']
        boolean = np.full([len(self.sampler)], False, dtype=bool)
        boolean[index] = True
        fp = np.memmap(os.path.join(self.opt.save_dir,label[self.counter]),dtype='double',mode='w+',shape=(len(index),2,1045))
        fp[:] = self.sampler[boolean,:,:]
        del fp
        newSampler = np.memmap(os.path.join(self.opt.save_dir,label[self.counter]),dtype='double',mode='r',shape=(len(index),2,1045))

        self.counter += 1 if self.counter % 2 != 0 else -1
        return {'A': np.asarray(newSampler).astype(np.float32)}#float)}

    def __getattr__(self, item):
        if item=='shape':
            raise ValueError("Not implemented")
            return self.shape
