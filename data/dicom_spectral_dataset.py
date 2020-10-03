import os
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as scp
# import torchvision.transforms as transforms

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.data_auxiliary import splitData, standardizeSpectra, normalizeSpectra
from util.util import progressbar
from util import util


class DicomSpectralDataset(BaseDataset):
    """
    DicomSpectralDataset compiles the spectroscopy data, processed using preprocess_UCSF.py, into a useable format for the network.
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        save_dir = os.path.join(self.opt.save_dir, 'data')
        util.mkdir(save_dir)
        
        if not self.opt.phase_data_path:
            self.split_npz(save_dir)
        
        path = os.path.join(self.opt.save_dir,'data/sizes')
        sizes = np.genfromtxt(path,delimiter=',').astype(np.int64)
        if (self.opt.phase=='train'):
            self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'train.dat'),dtype='double',mode='r',shape=(sizes[0],sizes[4],sizes[3]))
        elif (self.opt.phase=='val'):
            self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'val.dat'),dtype='double',mode='r',shape=(sizes[1],sizes[4],sizes[3]))
        elif (self.opt.phase=='test'):
            self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'test.dat'),dtype='double',mode='r',shape=(sizes[2],sizes[4],sizes[3]))
        self.counter=0
        print('Dataset sampler saved and loaded')

    def preprocess_numpy_spectra(self, spectraR, spectraI, size, save_dir):
        """
        Performs all necessary processing steps such as normalization and padding.
        """
        spectra = torch.empty([size[0], size[1], 2])

        # Preprocess and turn into torch.Tensor
        if self.opt.normalize:
            low, high = self.opt.norm_range
            maximum = torch.from_numpy(np.max([spectraR, spectraI], axis=1))
            minimum = torch.from_numpy(np.min([spectraR, spectraI], axis=1))
            # print('type(maximum), type(minimum): ',type(maximum), type(minimum))
            spectra[:,:,0] = normalizeSpectra(torch.from_numpy(spectraR),max=maximum[0,:],min=minimum[0,:],low=low, high=high)
            spectra[:,:,1] = normalizeSpectra(torch.from_numpy(spectraI),max=maximum[1,:],min=minimum[1,:],low=low, high=high)
            path = os.path.join(save_dir, 'min_and_max.mat')
            scp.savemat(path, mdict={'min': minimum, 'max': maximum})
        elif self.opt.standardize:
            S_real, S_imag = np.std(spectraR), np.std(spectraI)
            M_real, M_imag = np.mean(spectraR), np.mean(spectraI)
            spectra[:,:,0] = standardizeSpectra(torch.from_numpy(spectraR),mean=M_real,std=S_real)
            spectra[:,:,1] = standardizeSpectra(torch.from_numpy(spectraI),mean=M_imag,std=S_imag)
            path = os.path.join(save_dir, 'mean_and_std.mat')
            scp.savemat(path, mdict={'mean_real': M_real, 'mean_imag': M_imag, 'std_real': S_real, 'std_imag': S_imag})
        else:
            spectra[:,:,0] = torch.from_numpy(spectraR)
            spectra[:,:,1] = torch.from_numpy(spectraI)
        
        # # Padding
        if self.opt.pad_data > 0:
            print('spectra.size() before padding = ', spectra.size())
            spectra = F.pad(spectra, [0, 0, 0, self.opt.pad_data, 0, 0], "constant", 0) # 21
            print('spectra.size() after padding = ', spectra.size())

        if self.opt.input_nc==1:
            if self.opt.real==True:
                spectra = spectra[:,:,0].unsqueeze(dim=2)
            elif self.opt.imag==True:
                spectra = spectra[:,:,1].unsqueeze(dim=2)

        return spectra

    def split_npz(self, save_dir):
        """
        Loads the patient data from the disk, splits it into training, validation and test set and stores the respective sets as a .dat file.
        """
        spectraR = []
        spectraI = []
        L = []

        self.A_paths = make_dataset(self.root, file_type = 'numpy') # Returns a list of paths of the files in the dataset
        self.A_paths = sorted(self.A_paths)
        for i in progressbar(range(len(self.A_paths)), "Loading patient data: ", 20):
            datar, datai = np.load(self.A_paths[i]).get('data')
            spectraR.append(datar)
            spectraI.append(datai)
            L.append(len(spectraR[i]))
        print('len(spectraR) {}, len(spectraR[0]) {}'.format(len(spectraR), len(spectraR[0])))
        print('total number of imported spectra = ', sum(L))
        spectraR = np.concatenate(spectraR, axis=0)      # NOT axis=1!!!!
        print('spectraR.size() = ', spectraR.size, spectraR[0].size)
        spectraI = np.concatenate(spectraI, axis=0)      # NOT axis=1!!!!
        
        spectra = self.preprocess_numpy_spectra(spectraR, spectraI, spectraR.shape, save_dir)

        length, specLength, d = spectra.size()
        spectra = spectra.transpose(1, 2)
        self.shape = spectra.size()

        print('number of spectra: ',length)
        print('length of sepctra: ', specLength)
        print('spectra dimensionality: ',spectra.dim())

        print('----- Saving and Mapping Dataset ------')

        path = os.path.join(save_dir, 'Dataset')

        indices = [range(length)]
        np.save(path,spectra[indices])
        scp.savemat(path+'.mat',mdict={'spectra': np.asarray(spectra[indices,:,:])})
        print('Dataset saved as: {}'.format(path))

        # Split the data if indicated, save the indices in a CSV file
        if self.opt.split:# and not self.opt.k_folds:
            train_indices, val_indices, test_indices = splitData(length, self.opt.val_split, self.opt.test_split, self.opt.shuffle_data)
            contents = np.array([len(train_indices), len(val_indices), len(test_indices), specLength, d])
            path = os.path.join(self.opt.save_dir,'data/sizes')
            np.savetxt(path,contents,delimiter=',',fmt='%d')

            fp = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='double',mode='w+',shape=(len(train_indices),d,specLength))#length,specLength,2))
            fp[:] = spectra[train_indices]      # changed dtype from longdouble to double because it was returning dtype=np.float128
            del fp
            fp = np.memmap(os.path.join(self.opt.save_dir,'data/val.dat'),dtype='double',mode='w+',shape=(len(val_indices),d,specLength))#length,specLength,2))
            fp[:] = spectra[val_indices]      # changed dtype from longdouble to double because it was returning dtype=np.float128
            del fp
            fp = np.memmap(os.path.join(self.opt.save_dir,'data/test.dat'),dtype='double',mode='w+',shape=(len(test_indices),d,specLength))#length,specLength,2))
            fp[:] = spectra[test_indices]      # changed dtype from longdouble to double because it was returning dtype=np.float128
            del fp

        self.opt.phase_data_path = os.path.join(self.opt.save_dir,'data')


    def __getitem__(self, index):
        # 'Generates one sample of data'
        return {'A': np.asarray(self.sampler[index,:,:]).astype(float)}#np.float32)}#float)}

    def __len__(self):
        return len(self.sampler) # Determines the length of the dataloader

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
            return self.shape
