# Convert DCM file to matlab file. Calls the activatedSpectra.m file.
# Matlab file will later be loaded by the DataLoader.

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
import argparse
import matlab.engine
import numpy as np
import scipy.io as io
import torch
from util.util import progressbar
from util import util

print('--------------- Loading Options ---------------')
parser = argparse.ArgumentParser()
parser.add_argument('sourceDir', type=str, help='Directory of the DICOM files')
parser.add_argument('saveDir', type=str, default='./datasets/UCSF/', help='Directory where the matlab files will be saved')
parser.add_argument('file_ext_A', type=str, default='UCSF_proc.dcm',  help='File extension of the proc DICOM files')
parser.add_argument('file_ext_B', type=str, default='UCSF_NAA.dcm',  help='File extension of the NAA DICOM files')
parser.add_argument('folder_ext', type=str, default='UCSF_NAA.dcm',  help='Ending of the folder name where the UCSF DICOM files are located')


opt = parser.parse_args()

def make_dataset(dir, folder_ext, file_ext):
    """
    Returns a list of paths of the files in the given source directory
    """
    data = []
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    else:
        util.mkdir(dir)

    for root, d, fnames in sorted(os.walk(dir)):
        if root.endswith(folder_ext):
            for fname in fnames:
                if fname.endswith(file_ext):
                    path = os.path.join(root, fname)
                    data.append(path)
    return data

def convert_DCM_to_MAT(sourceDir: str, saveDir: str, file_ext_A: str, file_ext_B: str, folder_ext):
    print('>>>>>> Starting Matlab Engine... <<<<<<')
    eng = matlab.engine.start_matlab()
    print('>>>>>> Matlab Engine Running! <<<<<<')

    # Implementing Matlab code from UCSF
    # Compile loaded, reshaped data in row-wise matrix
    B_paths = sorted(make_dataset(sourceDir, folder_ext, file_ext_A))
    A_paths = sorted(make_dataset(sourceDir, folder_ext, file_ext_B))
    print('len(A_paths): ' + str(len(A_paths)) + ', len(B_paths): ' + len(B_paths))

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
        path = os.path.join(saveDir, basepath[1])
        util.mkdir(path)
        path1 = os.path.join(path, split[0])
        np.savez_compressed(path1,data=spectra)
        io.savemat(path1+'.mat',mdict={'spectra': np.array(spectra)})


convert_DCM_to_MAT(opt.sourceDir, opt.saveDir, opt.file_ext_A, opt.file_ext_B, opt.folder_ext)

print('vector of modes: ', compile)

print('Dataset saved in: ' + opt.saveDir)
