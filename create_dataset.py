"""
This script creates the dataset hirachy wich will later be used by the dataloader. 
For domain A and domain B, it transforms data in the following way:
.dcm ---> .mat / .npz ---> train.dat / val.dat

The resulting folder structure looks like this:
save_dir/
    spectra/
        train_A.dat
        train_B.dat
        val_A.dat
        val_B.dat
"""
import os.path
import matlab.engine
import numpy as np
import scipy.io as io
import torch
from util.util import progressbar

from data.image_folder import make_dataset
from data.data_auxiliary import splitData
from util.process_spectra import preprocess_numpy_spectra
from util.util import mkdir

from options.create_dataset_options import CreateDatasetOptions
opt = CreateDatasetOptions().parse()

def convert_DCM_to_MAT_NUMPY(sourceDir: str, file_ext_A: str, file_ext_B: str):
    """
    Loads the dicom encoded spectra and their respective metabolic maps, selects spectra of the activated voxels and stores them in a .npz and a .mat file
    """
    print('>>>>>> Starting Matlab Engine... <<<<<<')
    eng = matlab.engine.start_matlab()
    print('>>>>>> Matlab Engine Running! <<<<<<')

    # Implementing Matlab code from UCSF
    # Compile loaded, reshaped data in row-wise matrix
    A_paths = sorted(make_dataset(sourceDir, file_ext=file_ext_A))
    B_paths = sorted(make_dataset(sourceDir, file_ext=file_ext_B))
    print('Number of spectra: ' + str(len(A_paths)) + ', Number of metabolic maps: ' + str(len(B_paths)))

    for i in progressbar(range(len(A_paths)), "Processing patient data: ", 20):
        # Identify activated voxels using the NAA map and extract corresponding spectra
        dataR, dataI = eng.activatedSpectra(B_paths[i], A_paths[i], nargout=2)
        dataR, dataI = torch.FloatTensor(dataR), torch.FloatTensor(dataI)
        size = dataR.shape
        spectra = torch.empty([2, size[0], size[1]])

        spectra[0,:,:] = dataR
        spectra[1,:,:] = dataI
        # Store matlab + numpy file next to dicom files
        split = os.path.split(A_paths[i])
        path = os.path.join(split[0], os.path.splitext(split[1])[0])
        np.savez_compressed(path, data=spectra)
        io.savemat(path + '.mat', mdict={'spectra': np.array(spectra)})


def load_from_numpy(source_dir, save_dir):
    """
    Load samples from numpy files and perform preprocessing.
    Returns all spectra as a numpy array of size
    NUM_SAMPLES x 2 x SPECTRA_LENGTH
    """
    spectraR = []
    spectraI = []
    L = []

    A_paths = make_dataset(source_dir, file_type='numpy') # Returns a list of paths of the files in the dataset
    A_paths = sorted(A_paths)
    for i in progressbar(range(len(A_paths)), "Loading patient data: ", 20):
        datar, datai = np.load(A_paths[i]).get('data')
        spectraR.append(datar)
        spectraI.append(datai)
        L.append(len(spectraR[i]))
    spectraR = np.concatenate(spectraR, axis=0)
    spectraI = np.concatenate(spectraI, axis=0)
    
    spectra = preprocess_numpy_spectra(spectraR, spectraI, spectraR.shape, save_dir, opt)
    spectra = spectra.transpose(1, 2)
    return spectra

def load_from_mat(source_dir, save_dir, preprocess=False):
    """
    Load samples from matlab files. Note that there is no preprocessing step here!
    Returns all spectra as a numpy array of size
    NUM_SAMPLES x 2 x SPECTRA_LENGTH
    """
    spectra = []

    A_paths = make_dataset(source_dir, file_ext='spectra.mat') # Returns a list of paths of the files in the dataset
    A_paths = sorted(A_paths)
    for path in A_paths:
        spectra.append(io.loadmat(path)['spectra'])
    spectra = np.concatenate(spectra, axis=0)
    if preprocess:
        spectraR = spectra[:,1,:]
        spectraI = spectra[:,1,:]
        spectra = preprocess_numpy_spectra(spectraR, spectraI, spectraR.shape, save_dir)
    else:
        spectra = torch.from_numpy(spectra)
    return spectra


def split_dataset(spectra, save_dir, type):
    """
    Splits the dataset into training, validation and test set and saves them as .dat files.
    """
    print('total number of imported spectra = ', len(spectra))
    length, d, specLength= spectra.size()

    print('number of spectra: ',length)
    print('length of spectra: ', specLength)
    print('spectra dimensionality: ',spectra.shape)
    print('----- Saving and Mapping Dataset ------')

    path = os.path.join(save_dir, 'spectra')
    mkdir(path)

    # Split the data if indicated, save the indices in a CSV file
    if not opt.split:
        opt.val_split = opt.test_split = 0
    else:
        train_indices, val_indices, test_indices = splitData(length, opt.val_split, opt.test_split, opt.shuffle_data)
        contents = np.array([len(train_indices), len(val_indices), len(test_indices), specLength, d])
        np.savetxt(os.path.join(path,'sizes_'+type),contents,delimiter=',',fmt='%d')

        save_dat_file(os.path.join(path, 'train_' + type + '.dat'), train_indices, d, specLength, spectra)
        save_dat_file(os.path.join(path, 'val_' + type + '.dat'), val_indices, d, specLength, spectra)
        save_dat_file(os.path.join(path, 'test_' + type + '.dat'), test_indices, d, specLength, spectra)

def save_dat_file(path, indices, d, spec_length, spectra):
    """
    Saves all spectra of the given indices from spectra as .dat file under the given path.
    """
    if len(indices) > 0:
        fp = np.memmap(path, dtype='double',mode='w+',shape=(len(indices),d,spec_length))
        fp[:] = spectra[indices]
        del fp

def is_set_of_type(dir, type):
    """
    Checks if there is at least one .mat file under the given path (including subfolders)
    """
    for _, _, fnames in os.walk(dir):
        if any(fname.endswith(type) for fname in fnames):
            return True
    return False

def generate_dataset(type, source_dir):
    """
    This function creates the dataset hirachy wich will later be used by the dataloader. 
    For domain A and domain B, it transforms data in the following way:
    .dcm ---> .mat / .npz ---> train.dat / val.dat

    The resulting folder structure looks like this:
    save_dir/
        spectra/
            train_A.dat
            train_B.dat
            val_A.dat
            val_B.dat
    """
    if is_set_of_type(source_dir, '.dcm') and opt.force:
        convert_DCM_to_MAT_NUMPY(source_dir, opt.file_ext_spectra, opt.file_ext_metabolic_map)
        spectra = load_from_numpy(source_dir, opt.save_dir)
    elif is_set_of_type(source_dir, '.npz'):
        spectra = load_from_numpy(source_dir, opt.save_dir)
    elif is_set_of_type(source_dir, '.mat'):
        spectra = load_from_mat(source_dir, opt.save_dir, preprocess=False)
    elif is_set_of_type(source_dir, '.dcm'):
        convert_DCM_to_MAT_NUMPY(source_dir, opt.file_ext_spectra, opt.file_ext_metabolic_map)
        spectra = load_from_numpy(source_dir, opt.save_dir)
    else:
        raise ValueError("Source directory does not contain any valid spectra")
    split_dataset(spectra, opt.save_dir, type)


print('Generating dataset A...')
generate_dataset('A', opt.source_dir_A)
print('Generating dataset B...')
generate_dataset('B', opt.source_dir_B)
print('Done! You can find you dataset at', opt.save_dir)
