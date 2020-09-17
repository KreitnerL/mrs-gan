import os
import torch
import numpy as np
import torch.nn as nn

def splitData(opt, dataset_size, val_split=0.2, test_split=0.1, *both):
    if opt.shuffle_data:
        indices = torch.randperm(dataset_size)
    else:
        indices = range(dataset_size)
    split1 = torch.tensor([int(torch.floor((1 - torch.FloatTensor([val_split]) - torch.FloatTensor([test_split])) * dataset_size))])
    split2 = torch.tensor([int(torch.floor((1 - torch.FloatTensor([test_split])) * dataset_size))])    # split1 = torch.tensor([int(torch.floor((1 - torch.tensor(val_split, dtype=int) - torch.tensor(test_split)) * dataset_size))])

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


# def normalizeSpectraL2(input):



# def splitSpectra(spectra, size, opt):
#     length, d, specLength = size[0], size[1], size[2]
#     print('Splitting dataset into phases...')
#     print('     train: {}%, validate: {}%, test: {}%'.format(int((1-opt.val_split-opt.test_split)*100), int(opt.val_split*100), int(opt.test_split*100)))
#     train, val, test = splitData(opt, length, opt.val_split, opt.test_split)
#     path = os.path.join(opt.save_dir,'data/sizes')
#     contents = np.array([len(train), len(val), len(test), specLength, d])
#     np.savetxt(path,contents,delimiter=',',fmt='%d')
#     fp = np.memmap(os.path.join(opt.save_dir,'data/train.dat'),dtype='double',mode='w+',shape=(len(train),d,specLength))
#     fp[:] = spectra[train,:,:]
#     del fp
#     print('Train memory map saved')
#     if opt.val_split!=0:
#         fp = np.memmap(os.path.join(opt.save_dir,'data/val.dat'),dtype='double',mode='w+',shape=(len(val),d,specLength))
#         fp[:] = spectra[val,:,:]
#         del fp
#         print('Validation memory map saved')
#     if opt.test_split!=0:
#         fp = np.memmap(os.path.join(opt.save_dir,'data/test.dat'),dtype='double',mode='w+',shape=(len(test),d,specLength))
#         fp[:] = spectra[test,:,:]
#         del fp
#         print('Test memory map saved')
#     else:
#         print('Test memory map skipped')
#
#     return np.memmap(os.path.join(opt.save_dir,'data/train.dat'),dtype='double',mode='r',shape=(len(train),d,specLength))

def sample(dir,string,shape,tensor):
    fp = np.memmap(os.path.join(dir,string),dtype='float64',mode='w+',shape=shape)#length,specLength,2))
    fp[:] = tensor[:]; del fp
    return np.memmap(os.path.join(dir,string),dtype='float64',mode='r',shape=shape)
