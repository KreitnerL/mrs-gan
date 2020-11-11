import os
import torch
import numpy as np

def splitData(dataset_size, val_split=0.2, test_split=0.1, shuffle_data=False):
    """
    Divides the dataset into training, validation and test set.\n
    Returns the indices of samples for train, val and test set.
    """
    if shuffle_data:
        indices = torch.randperm(dataset_size)
    else:
        indices = torch.tensor(range(dataset_size))
    split1 = torch.tensor([round(dataset_size * (1 - val_split - test_split))])
    split2 = torch.tensor([round(dataset_size * (1 - test_split))])    # split1 = torch.tensor([int(torch.floor((1 - torch.tensor(val_split, dtype=int) - torch.tensor(test_split)) * dataset_size))])

    if not test_split==0:
        train_sampler, valid_sampler, test_sampler = indices[:split1], indices[split1:split2], indices[split2:]
    else:
        train_sampler, valid_sampler = indices[:split1], indices[split1:split2]
        test_sampler = torch.empty([0])

    train, _ = train_sampler.sort(dim=0)
    valid, _ = valid_sampler.sort(dim=0)
    test,  _ = test_sampler.sort(dim=0)

    return train, valid, test

def sample(dir,string,shape,tensor):
    fp = np.memmap(os.path.join(dir,string),dtype='float64',mode='w+',shape=shape)#length,specLength,2))
    fp[:] = tensor[:]; del fp
    return np.memmap(os.path.join(dir,string),dtype='float64',mode='r',shape=shape)
