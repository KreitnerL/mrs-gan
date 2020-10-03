import os
import os.path
import numpy as np
from torch import from_numpy

from data.base_dataset import BaseDataset

class DicomSpectralDataset(BaseDataset):
    """
    DicomSpectralDataset loads spectra from .dat files and returns a sample as a numpy array
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        path = os.path.join(self.opt.save_dir,'data/sizes')
        sizes_A = np.genfromtxt(os.path.join(self.opt.save_dir,'sizes_A') ,delimiter=',').astype(np.int64)
        sizes_B = np.genfromtxt(os.path.join(self.opt.save_dir,'sizes_B') ,delimiter=',').astype(np.int64)

        if (self.opt.phase=='train'):
            self.sampler_A = np.memmap(os.path.join(self.opt.phase_data_path,'train_A.dat'),dtype='double',mode='r',shape=(sizes_A[0],sizes_A[4],sizes_A[3]))
            self.sampler_B = np.memmap(os.path.join(self.opt.phase_data_path,'train_B.dat'),dtype='double',mode='r',shape=(sizes_B[0],sizes_B[4],sizes_B[3]))
        elif (self.opt.phase=='val'):
            self.sampler_A = np.memmap(os.path.join(self.opt.phase_data_path,'val_A.dat'),dtype='double',mode='r',shape=(sizes_A[1],sizes_A[4],sizes_A[3]))
            self.sampler_B = np.memmap(os.path.join(self.opt.phase_data_path,'val_B.dat'),dtype='double',mode='r',shape=(sizes_B[1],sizes_B[4],sizes_B[3]))
        elif (self.opt.phase=='test'):
            self.sampler_A = np.memmap(os.path.join(self.opt.phase_data_path,'test_A.dat'),dtype='double',mode='r',shape=(sizes_A[2],sizes_A[4],sizes_A[3]))
            self.sampler_B = np.memmap(os.path.join(self.opt.phase_data_path,'test_B.dat'),dtype='double',mode='r',shape=(sizes_B[2],sizes_B[4],sizes_B[3]))
        self.counter=0
        print('Dataset sampler loaded')

    def __getitem__(self, index):
        # 'Generates one sample of data'
        A = np.asarray(self.sampler_A[index,:,:]).astype(float)
        B = np.asarray(self.sampler_B[index,:,:]).astype(float)
        return {
            'A': from_numpy(A),
            'B': from_numpy(B)
        }

    def __len__(self):
        return max(len(self.sampler_A), len(self.sampler_B)) # Determines the length of the dataloader

    def name():
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
