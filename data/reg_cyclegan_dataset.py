import torch
from models.auxiliaries.physics_model_interface import PhysicsModel
from data.base_dataset import BaseDataset
import scipy.io as io
import numpy as np
from torch import from_numpy, empty
from util.util import normalize


class RegCycleGANDataset(BaseDataset):
    def initialize(self, opt, phase):
        self.phase = phase
        self.opt = opt
        self.roi = opt.roi
        self.root = opt.dataroot
        self.physics_model: PhysicsModel = opt.physics_model
        self.empty_tensor = empty(0)

        # Select relevant part of dataset
        if opt.representation == 'real':
            channel_index = slice(0,1)
        elif opt.representation == 'imag':
            channel_index = slice(1,2)
        else:
            channel_index = slice(None, None)
        if phase == 'train':
            self.selection = slice(0, opt.val_offset)
        elif phase == 'val':
            self.selection = slice(opt.val_offset, opt.test_offset)
        else:
            self.selection = slice(opt.test_offset, None)
        
        # Load dataset from .mat file
        all_data = io.loadmat(opt.dataroot)
        self.dataset = np.array(all_data[opt.dataname]).astype(float)
        self.innit_length(self.dataset.shape[-1])
        self.dataset = self.dataset[self.selection, channel_index, self.roi]
        if self.opt.representation == 'mag':
            self.dataset = np.expand_dims(np.sqrt(self.dataset[:,0,:]**2 + self.dataset[:,1,:]**2), 0)
        self.dataset = from_numpy(normalize(self.dataset))
        self.A_size = len(self.dataset)

        # Load labels from .mat file
        self.labels = []
        # if self.phase != 'test':
        for label_name in self.physics_model.get_label_names():
            if not label_name in all_data:
                print('WARNING: ' + label_name + ' not found in dataroot!')
                continue
            self.labels.append(all_data[label_name])
        self.num_labels = len(self.labels)
        self.labels = from_numpy(np.transpose(np.concatenate(self.labels, 0)))
        self.label_sampler = self.labels[self.selection]

    def generate_B_sample(self):
        param = torch.rand(self.num_labels)
        return param

    def innit_length(self, full_length):
        self.opt.full_data_length = full_length
        self.opt.data_length = len(range(0, full_length)[self.roi])

    def __getitem__(self, index):
        sample: dict = {
            'A': self.dataset[index % self.A_size],
            'label_A': self.label_sampler[index % self.A_size]
        }
        if self.phase == 'train':
            sample['B'] = self.generate_B_sample()
        return sample

    def __len__(self):
        return self.A_size

    def name(self):
        return 'Reg-CycleGAN-Dataset'