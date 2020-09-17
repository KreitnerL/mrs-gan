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








        #     # break
        #     # print('spectraI.shape = ',len(spectraI))
        #     # print('spectraI[0].shape = ',len(spectraI[0]))
        #     # print('Padding data...')
        #     # Real = F.pad(spectraR, (0, 21, 0, 0), "constant", 0)
        #     # Imag = F.pad(spectraI, (0, 21, 0, 0), "constant", 0)
        #     #
        #     # print('----------- Appended Data --------------')
        #     # print('spectraR.shape = ',len(Real))
        #     # print('spectraR[0].shape = ',len(Real[0]))
        #     # print('spectraI.shape = ',len(Imag))
        #     # print('spectraI[0].shape = ',len(Imag[0]))
        #     # # print('----------------------------------------')
        #
        #
        #     # print('')
        #     print('----- Saving and Mapping Dataset ------')
        #
        #     length = len(Real)
        #     print('len(Real) = ', len(Real))
        #     specLength = len(Real[1])
        #     mkdir(os.path.join(self.opt.save_dir,'data'))
        #     path = os.path.join(self.opt.save_dir, 'data/Dataset')
        #     size = Real.shape
        #     spectra = np.empty([size[0], size[1], 2])
        #     spectra[:,:,0] = Real
        #     spectra[:,:,1] = Imag
        #     print('Real and Imaginary components assigned')
        #     np.save(path,spectra)
        #     print('Dataset saved as: {}'.format(path))
        #
        #     # Split the data if indicated, save the indices in a CSV file, and set the sampler for the training phase
        #     # Assumption: if the dataset needs to be split, then chances are that the model has not been trained yet
        #     if self.opt.split:
        #         print('Splitting dataset into phases...')
        #         print('     train: {}%, validate: {}%, test: {}%'.format((1-self.opt.val_split+self.opt.test_split)*100,self.opt.val_split*100,self.opt.test_split*100))
        #         train, val, test = splitData(self.opt, length, self.opt.val_split, self.opt.test_split)
        #         path = os.path.join(self.opt.save_dir,'data/sizes')
        #         contents = np.array([len(train), len(val), len(test), specLength])
        #         print('contents = ', contents)
        #         np.savetxt(path,contents,delimiter=',',fmt='%d')
        #         fp = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='longdouble',mode='w+',shape=(len(train),specLength,2))
        #         # fp[:] = spectra[train.byte(),:,:]
        #         fp[:] = spectra[train,:,:]
        #         del fp
        #         print('Train memory map saved')
        #         fp = np.memmap(os.path.join(self.opt.save_dir,'data/val.dat'),dtype='longdouble',mode='w+',shape=(len(val),specLength,2))
        #         # fp[:] = spectra[val.byte(),:,:]
        #         fp[:] = spectra[val,:,:]
        #         del fp
        #         print('Validation memory map saved')
        #         if not self.opt.test_split==0:
        #             fp = np.memmap(os.path.join(self.opt.save_dir,'data/test.dat'),dtype='longdouble',mode='w+',shape=(len(test),specLength,2))
        #             # fp[:] = spectra[test.byte(),:,:]
        #             fp[:] = spectra[test,:,:]
        #             del fp
        #             print('Test memory map saved')
        #         else:
        #             print('Test memory map skipped')
        #
        #         self.sampler = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='longdouble',mode='w+',shape=(len(train),specLength,2))
        #         self.opt.phase_data_path = os.path.join(self.opt.save_dir,'data')
        #
        #     else:
        #         path = os.path.join(self.opt.save_dir,'data/sizes')
        #         contents = np.array([length, 0, 0, specLength])
        #         np.savetxt(path,contents,delimiter=',',fmt='%d')
        #         fp = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='longdouble',mode='w+',shape=(length,specLength,2))
        #         fp[:] = spectra[:]
        #         del fp
        #         self.sampler = np.memmap(os.path.join(self.opt.save_dir,'data/train.dat'),dtype='longdouble',mode='r',shape=(length,specLength,2))
        #
        # elif self.opt.phase_data_path:
        #     path = os.path.join(self.opt.save_dir,'data/sizes')
        #     sizes = np.genfromtxt(path,delimiter=',').astype(np.int64)
        #     if (self.opt.phase=='train'):
        #         self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'train.dat'),dtype='longdouble',mode='r',shape=(sizes[0],sizes[3],2))
        #     elif (self.opt.phase=='val'):
        #         self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'val.dat'),dtype='longdouble',mode='r',shape=(sizes[1],sizes[3],2))
        #     elif (self.opt.phase=='test'):
        #         self.sampler = np.memmap(os.path.join(self.opt.phase_data_path,'test.dat'),dtype='longdouble',mode='r',shape=(sizes[2],sizes[3],2))
        #
        # print('Dataset sampler saved and loaded')

#     def __getitem__(self, index):
#         # 'Generates one sample of data'
#         return {'A': np.asarray(self.sampler[index]).astype(float)}
#
#     def __len__(self):
#         return len(self.sampler) # Determines the length of the dataloader
#
    def name(self):
        return 'DicomSpectralDataset'
#
# def splitData(opt, length, val_split=0.2, test_split=0.1, *both):
#     dataset_size = length
#     indices = torch.randperm(dataset_size)
#     split1, split2 = torch.tensor([int(np.floor((1 - val_split - test_split) * dataset_size))]), torch.tensor([int(np.floor((1 - test_split) * dataset_size))])
#     if not test_split==0:
#         train_sampler, valid_sampler, test_sampler = indices[:split1], indices[split1:split2], indices[split2:]
#     else:
#         train_sampler, valid_sampler = indices[:split1], indices[split1:split2]
#         test_sampler = torch.empty([0])
#
#     # print('len(train_sampler), len(valid_sampler) = ', len(train_sampler), len(valid_sampler))
#     # print('len(train_sampler), len(valid_sampler), len(test_sampler) = ', len(train_sampler), len(valid_sampler), len(test_sampler))
#
#     train, _ = train_sampler.sort(dim=0)
#     valid, _ = valid_sampler.sort(dim=0)
#     test,  _ = test_sampler.sort(dim=0)
#
#     return train, valid, test
