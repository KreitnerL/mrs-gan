"""
This file contains various scripts that are used for preprocessing spectra before the are added to the dataset.
"""
import torch
import numpy as np
import scipy.io as io
import os
import torch.nn.functional as F

def standardizeSpectra(input, mean, std):
    return (input - mean) / std

def normalizeSpectra(input, max, min, low, high):
    mult = high - low
    out = ((input + max) / (max * 2)) * mult - (mult / 2)

    zeros = []
    for n in range(len(out)):
        if out[n,:].all()!=out[n,:].all(): # Check each spectra for matching as a whole before iterating through the flagged ones
            # Improves speed compared to iterating through all spectra to find inconsistencies
            for i in range(len(out[n,:])):
                if not out[n,i]==out[n,i]:
                    zeros.append(i)
                    out[n,i] = 0
    print('Number of NAN: ', len(zeros))

    return out

def preprocess_numpy_spectra(spectraR, spectraI, size, save_dir, opt):
    """
    Performs all necessary processing steps such as normalization and padding.
    """
    spectra = torch.empty([size[0], size[1], 2])

    # Preprocess and turn into torch.Tensor
    if opt.normalize:
        low, high = opt.norm_range
        maximum = torch.from_numpy(np.max([spectraR, spectraI], axis=1))
        minimum = torch.from_numpy(np.min([spectraR, spectraI], axis=1))
        # print('type(maximum), type(minimum): ',type(maximum), type(minimum))
        spectra[:,:,0] = normalizeSpectra(torch.from_numpy(spectraR),max=maximum[0,:],min=minimum[0,:],low=low, high=high)
        spectra[:,:,1] = normalizeSpectra(torch.from_numpy(spectraI),max=maximum[1,:],min=minimum[1,:],low=low, high=high)
        path = os.path.join(save_dir, 'min_and_max.mat')
        io.savemat(path, mdict={'min': minimum, 'max': maximum})
    elif opt.standardize:
        S_real, S_imag = np.std(spectraR), np.std(spectraI)
        M_real, M_imag = np.mean(spectraR), np.mean(spectraI)
        spectra[:,:,0] = standardizeSpectra(torch.from_numpy(spectraR),mean=M_real,std=S_real)
        spectra[:,:,1] = standardizeSpectra(torch.from_numpy(spectraI),mean=M_imag,std=S_imag)
        path = os.path.join(save_dir, 'mean_and_std.mat')
        io.savemat(path, mdict={'mean_real': M_real, 'mean_imag': M_imag, 'std_real': S_real, 'std_imag': S_imag})
    else:
        spectra[:,:,0] = torch.from_numpy(spectraR)
        spectra[:,:,1] = torch.from_numpy(spectraI)
    
    # # Padding
    if opt.pad_data > 0:
        print('spectra.size() before padding = ', spectra.size())
        spectra = F.pad(spectra, [0, 0, 0, opt.pad_data, 0, 0], "constant", 0) # 21
        print('spectra.size() after padding = ', spectra.size())

    if opt.input_nc==1:
        if opt.real==True:
            spectra = spectra[:,:,0].unsqueeze(dim=2)
        elif opt.imag==True:
            spectra = spectra[:,:,1].unsqueeze(dim=2)

    return spectra