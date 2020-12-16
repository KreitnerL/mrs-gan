import functools
import json
import os
import os.path
import numpy as np
from torch import from_numpy
from data.base_dataset import BaseDataset

index = {'train': 0, 'val': 1, 'test': 1}

class SpectraComponentDataset(BaseDataset):

    def name(self):
        return 'SpectraComponentDataset'

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

        if opt.phase == 'test':
            phase = 'val'
        else:
            phase = opt.phase

        sizes_A = np.genfromtxt(os.path.join(self.root,'sizes_A') ,delimiter=',').astype(np.int64)
        path_A = str(os.path.join(self.root, phase + '_A.dat'))
        path_B = str(os.path.join(self.root, phase + '_B.dat'))

        self.A_size = sizes_A[index[phase]]
        self.length = sizes_A[3]
        self.sampler_A = np.memmap(path_A, dtype='double', mode='r', shape=(self.A_size,sizes_A[4],sizes_A[3]))
        
        with open(path_B, 'r') as file:
            params:dict = json.load(file)
            self.sampler_B = np.transpose(list(params.values()))
            self.B_size = len(self.sampler_B)
        self.innit_transformations()

    def innit_transformations(self):
        self.transformations = [lambda A: np.asarray(A).astype(float)]
        if self.opt.mag:
            self.transformations.append(lambda A: np.expand_dims(np.sqrt(A[0,:]**2 + A[1,:]**2), 0))
        # self.transformations.append(lambda A: A/np.amax(abs(A)))
        self.transformations.append(lambda A: A/self.opt.relativator.detach().cpu().numpy())
        self.transformations.append(from_numpy)


    def __getitem__(self, index):
        # 'Generates one sample of data'
        if self.opt.phase != 'val':
            A = self.sampler_A[index % self.A_size,self.channel_index,self.roi]
            B = self.sampler_B[index % self.B_size]
            return {
                'A': self.transform(A),
                'B': from_numpy(B),
                'A_paths': '{:03d}.foo'.format(index % self.A_size),
                'B_paths': '{:03d}.foo'.format(index % self.B_size)
            }
        else:
            A = self.sampler_A[index % self.A_size,self.channel_index,self.roi]
            return {
                'A': self.transform(A),
                'A_paths': '{:03d}.foo'.format(index % self.A_size)
            }

    def __len__(self):
        if self.opt.phase != 'val':
            return max(self.A_size, self.B_size) # Determines the length of the dataloader
        else:
            return self.A_size

    def get_length(self):
        if self.opt.crop_start != None and self.opt.crop_end != None:
            l = self.opt.crop_end - self.opt.crop_start
        else:
            l = self.length
        # length must be power of 2
        assert (l & (l-1) == 0) and l != 0
        return l

    def transform(self, data):
        return functools.reduce((lambda x, y: y(x)), self.transformations, data)