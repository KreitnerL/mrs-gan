"""
This script creates the dataset hirachy wich will later be used by the dataloader. 
The resulting folder structure looks like this:
save_dir/
    spectra/
        train_A.dat
        train_B.dat
        val_A.dat
        val_B.dat
"""
import os.path
import numpy as np
import scipy.io as io
import torch
import json
from pathlib import Path

from data.image_folder import make_dataset
from data.data_auxiliary import splitData
from util.util import mkdir, normalize

from options.create_dataset_options import CreateDatasetOptions
opt = CreateDatasetOptions().parse()

labels = ["cho", "naa"]

def load_from_mat(source_dir, var_name):
    """
    Load samples from matlab files. Note that there is no preprocessing step here!
    Returns all spectra as a numpy array of size
    NUM_SAMPLES x 2 x SPECTRA_LENGTH
    """
    spectra = []
    filename = os.path.basename(source_dir)
    A_paths = make_dataset(Path(source_dir).parent, file_ext=filename) # Returns a list of paths of the files in the dataset
    A_paths = sorted(A_paths)
    for path in A_paths:
        spectra.append(io.loadmat(path)[var_name])
    spectra = np.concatenate(spectra, axis=0)
    spectra = normalize(spectra)
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

    path = os.path.join(save_dir, opt.name)
    mkdir(path)

    # Split the data if indicated, save the indices in a CSV file
    if not opt.split:
        opt.val_split = opt.test_split = 0
    else:
        train_indices, val_indices, test_indices = splitData(length, opt.val_split, opt.test_split, opt.shuffle_data)
        print("# training samples: {0}\n# validation samples: {1}\n# test samples: {2}".format(train_indices.size(0), val_indices.size(0), test_indices.size(0)))
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

def generate_dataset(type, source_dir, var_name, save_dir):
    """
    This function creates the dataset hirachy wich will later be used by the dataloader. 
    For domain A and domain B, it expects a matlab file containg all spectra in the form of [N x D x L]
    where N is the number of samples, D the number of dimensions and L the length of the spectra

    The resulting folder structure looks like this:
    save_dir/
        spectra/
            train_A.dat
            train_B.dat
            val_A.dat
            val_B.dat
    """
    spectra = load_from_mat(source_dir, var_name)
    split_dataset(spectra, save_dir, type)

def generate_labels(source_dir, labels, save_dir, val_split, test_split):
    """
    This function generates the labels for the validation of the target domain.
    """
    train_split = 1-val_split-test_split
    params = io.loadmat(source_dir)
    labels_dict = dict()
    for label in labels:
        p = np.squeeze(params[label])
        num_train = round(train_split*len(p))
        num_val = round(val_split*len(p))
        p = p[num_train : num_train+num_val]
        labels_dict[label] = p.tolist()
    with open(save_dir+'/labels.dat', 'w') as file:
        json.dump(labels_dict, file)

print('Generating dataset A...')
generate_dataset('A', opt.source_path_A, opt.A_mat_var_name, opt.save_dir)
print('Generating dataset B...')
generate_dataset('B', opt.source_path_B, opt.B_mat_var_name, opt.save_dir)
if len(opt.source_path_source_labels) and opt.val_split>0:
    print('Generating labels...')
    generate_labels(opt.source_path_source_labels, labels, opt.save_dir+opt.name, opt.val_split, opt.test_split)
print('Done! You can find you dataset at', opt.save_dir + opt.name + '/')
