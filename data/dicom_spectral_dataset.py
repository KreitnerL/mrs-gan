"""
DicomSpectralDataset compiles the spectroscopy data, processed using preprocess.py, into a useable format
for the SpectraGAN. This file has been

"""
import os
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as scp
# import torchvision.transforms as transforms

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.data_auxiliary import splitSpectra, standardizeSpectra, normalizeSpectra
from util.util import progressbar
from util import util


class DicomSpectralDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        datadir = os.path.join(self.opt.save_dir, 'data')
        util.mkdir(datadir)
        spectraR = []
        spectraI = []
        # Implementing Matlab code from UCSF
        # Compile loaded, reshaped data in row-wise matrix
        if not self.opt.phase_data_path:
            self.A_paths = make_dataset(self.root, self.opt) # Returns a list of paths of the files in the dataset
            self.A_paths = sorted(self.A_paths)
            for i in progressbar(range(len(self.A_paths)), "Loading patient data: ", 20):
                datar, datai = np.load(self.A_paths[i]).get('data')
                shape = datar.shape
                spectraR.append(datar)
                spectraI.append(datai)
            print('len(spectraR) {}, len(spectraR[0]) {}'.format(len(spectraR), len(spectraR[0])))
            L = []
            for i in range(len(spectraR)):
                L += [len(spectraR[i])]

            print('total number of imported spectra = ', sum(L))
            spectraR = np.concatenate(spectraR, axis=0)      # NOT axis=1!!!!
            print('spectraR.size() = ', spectraR.size, spectraR[0].size)
            spectraI = np.concatenate(spectraI, axis=0)      # NOT axis=1!!!!
            size = spectraR.shape
            spectra = torch.empty([size[0], size[1], 2])
            # print('spectra.size() = ', spectra.size())




            # if self.opt.normalize:
            #     low, high = self.opt.norm_range
            #     maximum = torch.from_numpy(np.max([spectraR, spectraI], axis=1))
            #     minimum = torch.from_numpy(np.min([spectraR, spectraI], axis=1))
            #     # print('type(maximum), type(minimum): ',type(maximum), type(minimum))
            #     spectra[:,:,0] = normalizeSpectra(torch.from_numpy(spectraR),max=maximum[0,:],min=minimum[0,:],low=low, high=high)
            #     spectra[:,:,1] = normalizeSpectra(torch.from_numpy(spectraI),max=maximum[1,:],min=minimum[1,:],low=low, high=high)
            #     path = os.path.join(datadir, 'min_and_max.mat')
            #     scp.savemat(path, mdict={'min': minimum, 'max': maximum})
            # elif self.opt.standardize:
            #     S_real, S_imag = np.std(spectraR), np.std(spectraI)
            #     M_real, M_imag = np.mean(spectraR), np.mean(spectraI)
            #     spectra[:,:,0] = standardizeSpectra(torch.from_numpy(spectraR),mean=M_real,std=S_real)
            #     spectra[:,:,1] = standardizeSpectra(torch.from_numpy(spectraI),mean=M_imag,std=S_imag)
            #     path = os.path.join(datadir, 'mean_and_std.mat')
            #     scp.savemat(path, mdict={'mean_real': M_real, 'mean_imag': M_imag, 'std_real': S_real, 'std_imag': S_imag})
            # else:
            spectra[:,:,0] = torch.from_numpy(spectraR)
            spectra[:,:,1] = torch.from_numpy(spectraI)

            # z = len(np.transpose(np.argwhere(spectra==0)))
            # print('Number of zeros in spectra: {}'.format(z))
            #
            # assert(spectra==spectra)

            print('spectra.size() before padding = ', spectra.size())
            # # Padding
            if opt.pad_data > 0:
                spectra = F.pad(spectra, [0, 0, 0, opt.pad_data, 0, 0], "constant", 0) # 21
                print('Padded spectra')
            # elif 'unet' in opt.which_model_netG:
            #     # no padding necessary
            #     pass

            print('spectra.size() after padding = ', spectra.size())

            if opt.input_nc==1:
                if opt.real==True:
                    spectra = spectra[:,:,0].unsqueeze(dim=2)
                elif opt.imag==True:
                    spectra = spectra[:,:,1].unsqueeze(dim=2)

            length, specLength, d = spectra.size()
            spectra = spectra.transpose(1, 2)
            self.shape = spectra.size()




            print('length, d, specLength: ', self.shape)
            print('------------ Loaded Data ---------------')
            print('spectra dimensionality: ',spectra.dim())
            print('number of spectra: ',length)#[0]))

            print('----- Saving and Mapping Dataset ------')

            path = os.path.join(datadir, 'Dataset')

            # indices = torch.randperm(length) if self.opt.shuffle==True else   # Keep the data in the same order when storing/saving
            indices = [range(length)]
            np.save(path,spectra[indices])
            scp.savemat(path+'.mat',mdict={'spectra': np.asarray(spectra[indices,:,:])})
            print('Dataset saved as: {}'.format(path))



            # Split the data if indicated, save the indices in a CSV file, and set the sampler for the training phase
            # Assumption: if the dataset needs to be split, then chances are that the model has not been trained yet
            if self.opt.split:# and not self.opt.k_folds:
                self.sampler = splitSpectra(spectra, spectra.size(), self.opt)
                self.opt.phase_data_path = os.path.join(self.opt.save_dir,'data')

            elif not self.opt.split:
                path = os.path.join(self.opt.save_dir,'data/sizes')
                contents = np.array([length, 0, 0, specLength, d])
                np.savetxt(path,contents,delimiter=',',fmt='%d')
                fp = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='double',mode='w+',shape=(length,d,specLength))#length,specLength,2))
                fp[:] = spectra[:]      # changed dtype from longdouble to double because it was returning dtype=np.float128
                del fp
                self.sampler = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='double',mode='r',shape=(length,d,specLength))#length,specLength,2))

        elif self.opt.phase_data_path:
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

    def __getitem__(self, index):
        # 'Generates one sample of data'
        return {'A': np.asarray(self.sampler[index,:,:]).astype(float)}#np.float32)}#float)}

    def __len__(self):
        return len(self.sampler) # Determines the length of the dataloader

    def name(self):
        return 'DicomSpectralDataset'

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
