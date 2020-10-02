import os
import torch
import numpy as np

def splitData(opt, dataset_size, val_split=0.2, test_split=0.1, *both):
    """
    Divides the dataset into training, validation and test set.\n
    Returns the indices of samples for train, val and test set.
    """
    if opt.shuffle_data:
        indices = torch.randperm(dataset_size)
    else:
        indices = range(dataset_size)
    split1 = torch.tensor([int(dataset_size * (1 - val_split - test_split))])
    split2 = torch.tensor([int(dataset_size * (1 - test_split))])    # split1 = torch.tensor([int(torch.floor((1 - torch.tensor(val_split, dtype=int) - torch.tensor(test_split)) * dataset_size))])

    if not test_split==0:
        train_sampler, valid_sampler, test_sampler = indices[:split1], indices[split1:split2], indices[split2:]
    else:
        train_sampler, valid_sampler = indices[:split1], indices[split1:split2]
        test_sampler = torch.empty([0])

    train, _ = train_sampler.sort(dim=0)
    valid, _ = valid_sampler.sort(dim=0)
    test,  _ = test_sampler.sort(dim=0)

    return train, valid, test

# def k_folds(length, folds, val_split=0.2, test_split=0.1):

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

def sample(dir,string,shape,tensor):
    fp = np.memmap(os.path.join(dir,string),dtype='float64',mode='w+',shape=shape)#length,specLength,2))
    fp[:] = tensor[:]; del fp
    return np.memmap(os.path.join(dir,string),dtype='float64',mode='r',shape=shape)
