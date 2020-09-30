"""
MatSpectralDataset compiles the spectroscopy data, processed using preprocess.py, into a usable format
for the SpectraGAN. This file has been

"""

# Todo: commit changes. Should be completely finalized for the current implementation

import nibabel as nib

import os
import os.path
import numpy as np
import torch
import scipy.io as io

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.data_auxiliary import standardizeSpectra, normalizeSpectra, splitData, sample
from util.util import progressbar
from util import util


class LabeledMatSpectralDataset(BaseDataset):
    def name(self):
        return 'LabeledMatSpectralDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        datadir = './dataset'#/debug'#os.path.join(self.opt.save_dir, 'data')
        util.mkdir(datadir)

        spec, mag, params = [], [], []

        if not self.opt.phase_data_path:
            self.A_paths = make_dataset(self.root, self.opt) # Returns a list of paths of the files in the dataset
            self.A_paths = sorted(self.A_paths)
            print('len(self.A_paths): ',len(self.A_paths))
            for i in progressbar(range(len(self.A_paths)), "Loading patient data: ", 20):
                if 'dataset_parameters.mat' in self.A_paths[i]:
                    params = io.loadmat(self.A_paths[i])
                    # print(params.keys())
                elif 'dataset_magnitude.mat' in self.A_paths[i]:
                    mag = io.loadmat(self.A_paths[i])
                    # print(mag.keys())
                elif 'dataset_spectra.mat' in self.A_paths[i]:
                    spec = io.loadmat(self.A_paths[i])
                    # print(spec.keys())

            data = spec['spectra']
            mag = mag['mag']
            cho = np.transpose(params['cho'])
            cre = np.transpose(params['cre'])
            naa = np.transpose(params['naa'])
            glx = np.transpose(params['glx'])
            ins = np.transpose(params['ins'])
            mac = np.transpose(params['mac'])
            lip = np.transpose(params['lip'])
            t2 = params['t2']
            freq_shift = params['freq_shift']
            noise = np.transpose(params['noise'])
            phase = params['phase']
            base_scale = np.moveaxis(np.asarray([params['base_scale0'], params['base_scale1'], params['base_scale2'],
                                     params['base_scale3'], params['base_scale4']]), 2, 0)
            print('Total number of imported spectra = ', len(mag))

            # data = np.moveaxis(data, 1, -1)

            size = data.shape
            print('data.shape: ',size)
            spectra = np.ndarray([size[0], size[1], 2]) # [size[0], size[1], 2]
            # spectra[:,0,:] = data[:,0,:]
            # spectra[:,1,:] = data[:,1,:]
            spectra = data
            size = mag.shape
            # magnitude = np.ndarray([size[0], size[2], 1])
            # magnitude = np.moveaxis(mag, 1, -1)# [:,:,0]

            magnitude = mag
            # Define parameters vector
            # print(cho.shape)
            parameters = np.empty([size[0],23,1])
            # print('parameters.shape: ', parameters.shape)
            # print(cho.shape, cre.shape, naa.shape, glx.shape, ins.shape, mac.shape,lip.shape)
            # print('t2.shape: ', t2.shape)
            # print('freq_shift.shape: ', freq_shift.shape)
            # print(noise.shape, phase.shape, base_scale.shape)
            parameters[:,0,:] = cho
            parameters[:,1,:] = cre#.transpose(1,0)
            parameters[:,2,:] = naa
            parameters[:,3,:] = glx
            parameters[:,4,:] = ins
            parameters[:,5,:] = mac
            parameters[:,6,:] = lip
            parameters[:,7,:] = np.expand_dims(t2[:,0],axis=-1)
            parameters[:,8,:] = np.expand_dims(t2[:,1],axis=-1)
            parameters[:,9,:] = np.expand_dims(t2[:,2],axis=-1)
            parameters[:,10,:] = np.expand_dims(t2[:,3],axis=-1)
            parameters[:,11,:] = np.expand_dims(t2[:,4],axis=-1)
            parameters[:,12,:] = np.expand_dims(t2[:,5],axis=-1)
            parameters[:,13,:] = np.expand_dims(t2[:,6],axis=-1)
            parameters[:,14,:] = np.expand_dims(freq_shift,axis=-1)
            parameters[:,15,:] = noise#np.expand_dims(noise,axis=-1)
            parameters[:,16,:] = np.expand_dims(phase[:,0],axis=1)
            parameters[:,17,:] = np.expand_dims(phase[:,1],axis=1)
            parameters[:,18,:] = base_scale[:,0,:]
            parameters[:,19,:] = base_scale[:,1,:]
            parameters[:,20,:] = base_scale[:,2,:]
            parameters[:,21,:] = base_scale[:,3,:]
            parameters[:,22,:] = base_scale[:,4,:]
            del cho, cre, naa, glx, ins, mac, lip, freq_shift, t2, noise, phase, base_scale, data, mag, params, spec

            if self.opt.normalize:
                print('Normalizing spectra...')
                low, high = self.opt.norm_range
                maximum = np.amax([spectra[:,0,:], spectra[:,1,:]])
                minimum = np.amin([spectra[:,0,:], spectra[:,1,:]])
                spectra[:,0,:] = normalizeSpectra(spectra[:,0,:],max=maximum,min=minimum,low=low, high=high)
                spectra[:,1,:] = normalizeSpectra(spectra[:,1,:],max=maximum,min=minimum,low=low, high=high)
                path = os.path.join(datadir, 'min_and_max_spectra.mat')
                io.savemat(path, mdict={'min': minimum, 'max': maximum})

                maximum = np.amax(magnitude)
                minimum = np.amin(magnitude)
                magnitude = normalizeSpectra(magnitude,max=maximum,min=minimum,low=low, high=high)
                path = os.path.join(datadir, 'min_and_max_magnitude.mat')
                io.savemat(path, mdict={'min': minimum, 'max': maximum})

            elif self.opt.standardize:
                print('Standardizing spectra...')
                S_real, S_imag = np.std(spectra[:,:,0]), np.std(spectra[:,1,:])
                M_real, M_imag = np.mean(spectra[:,:,0]), np.mean(spectra[:,1,:])
                spectra[:,0,:] = standardizeSpectra(torch.from_numpy(spectra[:,0,:]),mean=M_real,std=S_real)
                spectra[:,1,:] = standardizeSpectra(torch.from_numpy(spectra[:,1,:]),mean=M_imag,std=S_imag)
                path = os.path.join(datadir, 'mean_and_std_spectra.mat')
                io.savemat(path, mdict={'mean_real': M_real, 'mean_imag': M_imag, 'std_real': S_real, 'std_imag': S_imag})

                S_mag = np.std(magnitude)
                M_mag = np.mean(magnitude)
                magnitude = standardizeSpectra(torch.from_numpy(magnitude),mean=M_mag,std=S_mag)
                path = os.path.join(datadir, 'mean_and_std_magnitude.mat')
                io.savemat(path, mdict={'mean_magnitude': M_mag, 'std_magnitude': S_mag})

            if opt.input_nc==1:
                if opt.real==True:
                    spectra = spectra[:,0,:].unsqueeze(dim=1)
                elif opt.imag==True:
                    spectra = spectra[:,1,:].unsqueeze(dim=1)

            length, d, specLength = spectra.shape
            spectra = np.asarray(spectra,dtype='float64')#.transpose(0,2,1)
            magnitude = np.asarray(magnitude,dtype='float64')#.transpose(0,2,1)
            parameters = np.asarray(parameters.transpose(0,2,1),dtype='float64')
            self.shape_spec = spectra.shape
            self.shape_mag = magnitude.shape
            self.shape_params = parameters.shape
            print('self.shape_params = ', self.shape_params)

            print('length, d, specLength: ', self.shape)
            print('------------- Loaded Data -------------')
            print('spectra dimensionality: ',spectra.ndim)
            print('number of spectra: ',length)

            print('----- Saving and Mapping Dataset ------')
            if opt.split and not self.opt.phase_data_path:
                print('Splitting dataset into phases...')
                print('     train: {}%, validate: {}%, test: {}%'.format(int((1-opt.val_split-opt.test_split)*100), int(opt.val_split*100), int(opt.test_split*100)))
                train, val, test = splitData(opt, length, opt.val_split, opt.test_split)
                path = os.path.join(datadir,'sizes.txt')
                contents = np.array([len(train), len(val), len(test), specLength, d, self.shape_mag[1], self.shape_params[2]])
                np.savetxt(path,contents,delimiter=',',fmt='%d')

                # # Complex Spectra
                if opt.test_split > 0.:
                    _ = sample(datadir,'test_spectra.dat',(len(test),d,specLength),spectra[test,:,:])
                _ = sample(datadir,'val_spectra.dat',(len(val),d,specLength),spectra[val,:,:])
                self.sampler_spectra = sample(datadir,'train_spectra.dat',(len(train),d,specLength),spectra[train,:,:])

                # # Magnitude Spectra
                if opt.test_split > 0.:
                    _ = sample(datadir,'test_magnitude.dat',(len(test),self.shape_mag[1],self.shape_mag[2]),magnitude[test,:,:])
                _ = sample(datadir,'val_magnitude.dat',(len(val),self.shape_mag[1],self.shape_mag[2]),magnitude[val,:,:])
                self.sampler_mag = sample(datadir,'train_magnitude.dat',(len(train),self.shape_mag[1],self.shape_mag[2]),magnitude[train,:,:])

                # # Parameters
                if opt.test_split > 0.:
                    _ = sample(datadir,'test_parameters.dat',(len(test),self.shape_params[1],self.shape_params[2]),parameters[test,:,:])
                _ = sample(datadir,'val_parameters.dat',(len(val),self.shape_params[1],self.shape_params[2]),parameters[val,:,:])
                self.sampler_params = sample(datadir,'train_parameters.dat',(len(train),self.shape_params[1],self.shape_params[2]),parameters[train,:,:])

                self.length = contents[0:2]
                self.opt.phase_data_path = datadir

        else:#elif self.opt.phase_data_path:
            assert os.path.isdir(self.opt.phase_data_path), '%s is not a valid path' % self.opt.phase_data_path
            print('phase_data_path: ',datadir)
            print('phase: ',self.opt.phase)
            path = os.path.join(self.opt.phase_data_path,'sizes.txt')
            sizes = np.genfromtxt(path,delimiter=',').astype(np.int64)
            self.length = sizes[0:2].astype(np.int64)
            if (self.opt.phase=='train'):
                print('sizes: ',sizes)
                self.sampler_spectra = np.memmap(os.path.join(self.opt.phase_data_path,'train_spectra.dat'),dtype='float64',mode='r',shape=(sizes[0],sizes[4],sizes[3]))
                self.sampler_mag = np.memmap(os.path.join(self.opt.phase_data_path,'train_magnitude.dat'),dtype='float64',mode='r',shape=(sizes[0],sizes[5],sizes[3]))
                self.sampler_params = np.memmap(os.path.join(self.opt.phase_data_path,'train_parameters.dat'),dtype='float64',mode='r',shape=(sizes[0],sizes[5],sizes[6]))
                print('>>> Training data loaded.')
            elif (self.opt.phase=='val'):
                self.sampler_spectra = np.memmap(os.path.join(self.opt.phase_data_path,'val_spectra.dat'),dtype='float64',mode='r',shape=(sizes[1],sizes[4],sizes[3]))
                self.sampler_mag = np.memmap(os.path.join(self.opt.phase_data_path,'val_magnitude.dat'),dtype='float64',mode='r',shape=(sizes[1],sizes[5],sizes[3]))
                self.sampler_params = np.memmap(os.path.join(self.opt.phase_data_path,'val_parameters.dat'),dtype='float64',mode='r',shape=(sizes[1],sizes[5],sizes[6]))
                print('>>> Validation data loaded.')
            elif (self.opt.phase=='test'):
                self.sampler_spectra = np.memmap(os.path.join(self.opt.phase_data_path,'test_spectra.dat'),dtype='float64',mode='r',shape=(sizes[2],sizes[4],sizes[3]))
                self.sampler_mag = np.memmap(os.path.join(self.opt.phase_data_path,'test_magnitude.dat'),dtype='float64',mode='r',shape=(sizes[2],sizes[5],sizes[3]))
                self.sampler_params = np.memmap(os.path.join(self.opt.phase_data_path,'test_parameters.dat'),dtype='float64',mode='r',shape=(sizes[2],sizes[5],sizes[6]))
                print('>>> Testing data loaded.')
            self.length = sizes[0:2].astype(np.int64)

        assert os.path.isfile(os.path.join(datadir,'sizes.txt')), 'sizes.txt has not been saved'

        print('Dataset samplers saved and loaded')

    def __getitem__(self, index):
        # 'Generates one sample of data'
        return {'spectra': torch.from_numpy(self.sampler_spectra[index,:,:]).float(),
                'magnitude': torch.from_numpy(self.sampler_mag[index,:,:]).float(),
                'params': torch.from_numpy(self.sampler_params[index,:,:]).float().transpose(-1,-2)}

    def __len__(self):
        # path = os.path.join(self.opt.save_dir,'data/sizes.txt')
        path = './dataset/sizes.txt'
        sizes = np.genfromtxt(path,delimiter=',').astype(np.int64)

        if self.phase=='train':
            return sizes[0] # Determines the length of the dataloader
        elif self.phase=='val':
            return sizes[1]
        elif self.phase=='test':
            return sizes[2]

    def __getattr__(self, item):
        if item=='shape':
            return self.shape_spec
        elif item=='magshape':
            return self.shape_mag
