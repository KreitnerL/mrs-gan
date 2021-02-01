import functools
import json
import os
import os.path
import numpy as np
from torch import from_numpy, empty
from data.base_dataset import BaseDataset

index = {'train': 0, 'val': 1, 'test': 1}

class SpectraComponentDataset(BaseDataset):

    def name(self):
        return 'SpectraComponentDataset'

    def initialize(self, opt, phase):
        self.phase = phase
        # Setup
        self.opt = opt
        self.roi = self.opt.roi
        self.root = opt.dataroot
        if opt.real:
            self.channel_index = slice(0,1)
        elif opt.imag:
            self.channel_index = slice(1,2)
        else:
            self.channel_index = slice(None, None)

        # Load data
        sizes_A = np.genfromtxt(os.path.join(self.root,'sizes_A') ,delimiter=',').astype(np.int64)
        path_A = str(os.path.join(self.root, phase + '_A.dat'))
        path_labels_A = str(os.path.join(self.root, phase + '_labels_A.dat'))
        path_B = str(os.path.join(self.root, phase + '_B.dat'))

        self.A_size = sizes_A[index[phase]]
        self.length = sizes_A[3]
        self.sampler_A = np.memmap(path_A, dtype='double', mode='r', shape=(self.A_size,sizes_A[4],sizes_A[3]))

        if os.path.isfile(path_labels_A):
            with open(path_labels_A, 'r') as file:
                params:dict = json.load(file)
                self.opt.label_names = list(params.keys())
                self.sampler_labels_A =  from_numpy(np.transpose(list(params.values())))
        else:
            self.sampler_labels_A = None
            self.empty_tensor = empty(0)
        
        if self.phase == 'train':
            with open(path_B, 'r') as file:
                params:dict = json.load(file)
                self.sampler_B = from_numpy(np.transpose(list(params.values())))
                self.B_size = len(self.sampler_B)

        self.innit_transformations()
        self.innit_length()

    def innit_transformations(self):
        self.transformations = [lambda A: np.asarray(A).astype(float)]
        if self.opt.mag:
            self.transformations.append(lambda A: np.expand_dims(np.sqrt(A[0,:]**2 + A[1,:]**2), 0))
        self.transformations.append(lambda A: A/np.amax(abs(A)))
        # self.transformations.append(lambda A: A/self.opt.relativator.detach().cpu().numpy())
        self.transformations.append(from_numpy)


    def __getitem__(self, index):
        # 'Generates one sample of data'
        if self.phase == 'train':
            A = self.transform(self.sampler_A[index % self.A_size,self.channel_index,self.roi])
            label_A = self.sampler_labels_A[index % self.A_size] if self.sampler_labels_A is not None else self.empty_tensor
            B = self.sampler_B[index % self.B_size]
            return {
                'A': A,
                'label_A': label_A,
                'B': B,
                'A_paths': '{:03d}.foo'.format(index % self.A_size),
                'B_paths': '{:03d}.foo'.format(index % self.B_size)
            }
        else:
            A = self.sampler_A[index % self.A_size,self.channel_index,self.roi]
            label_A = self.sampler_labels_A[index % self.A_size] if self.sampler_labels_A is not None else self.empty_tensor
            return {
                'A': self.transform(A),
                'label_A': label_A,
                'A_paths': '{:03d}.foo'.format(index % self.A_size)
            }

    def __len__(self):
        if self.phase == 'train' and self.opt.phase == 'train':
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def innit_length(self):
        self.opt.full_data_length = self.length
        self.opt.data_length = len(range(0, self.length)[self.roi])

    def transform(self, data):
        return functools.reduce((lambda x, y: y(x)), self.transformations, data)