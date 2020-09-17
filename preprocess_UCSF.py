"""
This file sorts through the hierarchy of the UCSF dataset I received.
It looks for the UCSF data (vs LCM data) and uses the pre-processed, baseline-normalized spectra ('*proc.dcm'),
converts the spectra into a list-wise matrix, removes the spectra from the non-activated voxels, before saving
the list of spectra into a compressed numpy file.

To adjust the hierarchy structure for parsing different datasets, modify the make_dataset function.

Use the checkpoints_dir input to specify the save directory.

To run this code, use the preprocess_spectra_dataset.sh file from teh repository's main directory.
"""

import os.path
import time

import matlab.engine
import numpy as np
import pydicom
import scipy.io as io
import torch
import torch.nn.functional as F
from models.auxiliary import progressbar
from options.train_options import TrainOptions
from torch import empty, flip, reshape, squeeze, transpose
from util import util
from util.util import mkdir

print('--------------- Loading Options ---------------')
opt = TrainOptions().parse()
root = opt.dataroot
dir_A = os.path.join(opt.dataroot, opt.phase)
savedir = './datasets/'#opt.chekcpoints_dir
mkdir(savedir)
opt.file_ext = 'UCSF_proc.dcm'
opt.folder_ext = 'ucsf'

print('>>>>>> Starting Matlab Engine <<<<<<')
eng = matlab.engine.start_matlab()
print('>>>>>> Completed <<<<<<')

def make_dataset(dir, opt):
    data = []
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    else:
        util.mkdir(dir)

    for root, d, fnames in sorted(os.walk(dir)):
        if root.endswith(opt.folder_ext):
            for fname in fnames:
                if fname.endswith(opt.file_ext):
                    path = os.path.join(root, fname)
                    data.append(path)
    return data

# Implementing Matlab code from UCSF
# Compile loaded, reshaped data in row-wise matrix
B_paths = sorted(make_dataset(root, opt)) # Returns a list of paths of the files in the dataset
opt.file_ext = 'UCSF_NAA.dcm'
A_paths = sorted(make_dataset(root, opt))
print('len(A_paths), len(B_paths) = ', len(A_paths), len(B_paths))

print('>>>>>>> Starting <<<<<<<')
for i in progressbar(range(len(B_paths)), "Processing patient data: ", 20):
    # Identify activated voxels using the NAA map and extract corresponding spectra
    dataR, dataI = eng.activatedSpectra(A_paths[i], B_paths[i], nargout=2)

    dataR, dataI = torch.FloatTensor(dataR), torch.FloatTensor(dataI)

    # real = F.pad(dataR, (0, 21, 0, 0), "constant", 0)
    # imag = F.pad(dataI, (0, 21, 0, 0), "constant", 0)
    size = dataR.shape
    spectra = torch.empty([2, size[0], size[1]])

    spectra[0,:,:] = dataR
    spectra[1,:,:] = dataI
    split1 = os.path.split(B_paths[i])
    basepath = os.path.split(os.path.split(split1[0])[0])
    split = os.path.splitext(split1[1])
    path = os.path.join(savedir, basepath[1])
    util.mkdir(path)
    path1 = os.path.join(path, split[0])
    np.savez_compressed(path1,data=spectra)
    io.savemat(path1+'.mat',mdict={'spectra': np.array(spectra)})

print('vector of modes: ', compile)

print('Dataset saved in: {}'.format(os.path.split(path)[0]))
