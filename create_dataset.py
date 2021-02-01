"""
This script creates the dataset hirachy wich will later be used by the dataloader. 
The resulting folder structure looks like this:
save_dir/
    name/
        train_A.dat
        train_B.dat
        val_A.dat
        val_B.dat
"""
import os
import os.path
import numpy as np
import scipy.io as io
import torch
import json
from pathlib import Path

from data.image_folder import make_dataset
from util.util import mkdir

from options.create_dataset_options import CreateDatasetOptions
opt = CreateDatasetOptions().parse()

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
    # spectra = normalize(spectra)
    spectra = torch.from_numpy(spectra)
    return spectra

def set_split_indices(dataset_size, val_split=0.2, test_split=0.1, shuffle_data=False):
    """
    Divides the dataset into training, validation and test set.\n
    Returns the indices of samples for train, val and test set.
    """
    if not opt.split:
        opt.train_indices, opt.val_indices, opt.test_indices = np.array(range(dataset_size)), np.empty(0), np.empty(0)
        return

    if shuffle_data:
        indices = np.random.permutation(dataset_size)
    else:
        indices = np.array(range(dataset_size))
    split1 = round(dataset_size * (1 - val_split - test_split))
    split2 = round(dataset_size * (1 - test_split))    # split1 = torch.tensor([int(torch.floor((1 - torch.tensor(val_split, dtype=int) - torch.tensor(test_split)) * dataset_size))])

    if not test_split==0:
        train_sampler, valid_sampler, test_sampler = indices[:split1], indices[split1:split2], indices[split2:]
    else:
        train_sampler, valid_sampler = indices[:split1], indices[split1:split2]
        test_sampler = np.empty(0)

    
    print("# training samples: {0}\n# validation samples: {1}\n# test samples: {2}".format(len(train_sampler), len(valid_sampler), len(test_sampler)))

    opt.train_indices = np.sort(train_sampler, axis=0)
    opt.val_indices = np.sort(valid_sampler, axis=0)
    opt.test_indices = np.sort(test_sampler, axis=0)


def split_dataset(spectra, labels, save_dir, type):
    """
    Splits the dataset into training, validation and test set and saves them as .dat files.
    """
    length, d, specLength = spectra.size()
    print('Total number of samples:', len(spectra))
    print('spectra dimensionality: ',spectra.shape)
    mkdir(save_dir)

    if all(x is None for x in (opt.train_indices, opt.val_indices, opt.test_indices)):
        set_split_indices(length, opt.val_split, opt.test_split, opt.shuffle_data)

    contents = np.array([len(opt.train_indices), len(opt.val_indices), len(opt.test_indices), specLength, d])
    np.savetxt(os.path.join(save_dir,'sizes_'+type),contents,delimiter=',',fmt='%d')
    save_dat_file(os.path.join(save_dir, 'train_' + type + '.dat'), opt.train_indices, d, specLength, spectra)
    save_dat_file(os.path.join(save_dir, 'val_' + type + '.dat'), opt.val_indices, d, specLength, spectra)
    save_dat_file(os.path.join(save_dir, 'test_' + type + '.dat'), opt.test_indices, d, specLength, spectra)

    if labels and len(opt.train_indices) > 0:
        with open(save_dir+'/train_labels_A.dat', 'w') as file:
            json.dump({key: val[opt.train_indices].tolist() for key,val in labels.items()}, file)
    if labels and len( opt.val_indices) > 0:
        with open(save_dir+'/val_labels_A.dat', 'w') as file:
            json.dump({key: val[opt.val_indices].tolist() for key,val in labels.items()}, file)
    if labels and len( opt.test_indices) > 0:
        with open(save_dir+'/test_labels_A.dat', 'w') as file:
            json.dump({key: val[opt.test_indices].tolist() for key,val in labels.items()}, file)
            

def save_dat_file(path, indices, d, spec_length, spectra):
    """
    Saves all spectra of the given indices from spectra as .dat file under the given path.
    """
    if len(indices) > 0:
        fp = np.memmap(path, dtype='double',mode='w+',shape=(len(indices),d,spec_length))
        fp[:] = spectra[indices]
        del fp

def generate_dataset(type, spectra_path, var_name, labels_path, label_names, save_dir):
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
    spectra = load_from_mat(spectra_path, var_name)
    if labels_path:
        labels_dict = io.loadmat(labels_path)
        labels = {key: np.squeeze(labels_dict[key]) for key in label_names}
    else:
        labels = None
    split_dataset(spectra, labels, save_dir, type)

def generate_quantity_dataset(type, source_dir, save_dir, label_names):
    params = io.loadmat(source_dir)
    labels_train = dict()
    for label in label_names:
        p = np.squeeze(params[label])
        labels_train[label] = p.tolist()

    print("# training samples: {0}\n# validation samples: {1}\n# test samples: {2}".format(len(labels_train[label_names[0]]), 0, 0))
    base, _ = os.path.split(save_dir)
    os.makedirs(base,exist_ok=True)
    with open(save_dir+'train_%s.dat'%type, 'w') as file:
        json.dump(labels_train, file)

if __name__ == "__main__":
    print('Generating dataset A...')
    generate_dataset('A', opt.source_path_A, opt.A_mat_var_name, opt.source_path_source_labels, opt.label_names, opt.save_dir)
    print('Generating dataset B...')
    generate_quantity_dataset('B', opt.source_path_B, opt.save_dir, opt.label_names)

    print('Done! You can find you dataset at', opt.save_dir)
