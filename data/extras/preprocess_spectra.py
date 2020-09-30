import torch
import os.path
import numpy as np
import pydicom
import torch.nn.functional as F
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from util.util import progressbar
from util.util import mkdir


class PreprocessDicomSpectralDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        counter = 0
        spectraR = []
        spectraI = []
        # Implementing Matlab code from UCSF
        # Compile loaded, reshaped data in row-wise matrix
        # if not self.opt.phase_data_path:
        self.A_paths = sorted(make_dataset(self.root, self.opt)) # Returns a list of paths of the files in the dataset
        savedir = '~/datasets/spectra'
        mkdir(savedir)#os.path.join('./datasets','spectra'))
        for i in progressbar(range(len(self.A_paths)), "Loading patient data: ", 20):
            with pydicom.filereader.dcmread(self.A_paths[i]) as file:
                split1 = os.path.split(self.A_paths[1])
                split = os.path.splitext(split1[1])
                data = np.array(file.SpectroscopyData)
                temp = data.reshape([2,file.DataPointColumns,file.Rows,file.Columns,file.NumberOfFrames])
                tempR = np.squeeze(temp[0,:,:,:])
                tempI = np.squeeze(temp[1,:,:,:])
                specR = np.flip(np.flip(np.transpose(tempR,[0,2,1,3]),1),2)
                specR = np.reshape(specR,[-1,file.DataPointColumns])
                specI = np.flip(np.flip(np.transpose(tempI,[0,2,1,3]),1),2)
                specI = np.reshape(specI,[-1,file.DataPointColumns])

                spectraR = torch.FloatTensor(np.concatenate(specR,axis=0))#.tolist()
                spectraI = torch.FloatTensor(np.concatenate(specI,axis=0))#.tolist()
                for i in range(len(spectraR)):
                    if sum(spectraR[i])>0:
                        Real = F.pad(spectraR, (0, 21, 0, 0), "constant", 0)
                        Imag = F.pad(spectraI, (0, 21, 0, 0), "constant", 0)
                        path = os.path.join('./datasets','spectra', split[1])
                        size = Real.shape
                        spectra = np.empty([size[0], size[1], 2])
                        spectra[:,:,0] = Real
                        spectra[:,:,1] = Imag
                        # print('Real and Imaginary components assigned')
                        np.save(path,spectra)


    # def __getitem__(self, index):
    #     # 'Generates one sample of data'
    #     return {'A': np.asarray(self.sampler[index]).astype(float)}

    def __len__(self):
        return len(self.A_paths) # Determines the length of the dataloader

    def name(self):
        return 'PreprocessDicomSpectralDataset'
