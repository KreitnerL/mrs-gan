import torch
import os
import numpy as np
import scipy.io as io

if __name__ == "__main__":
    totalEntries=100000
    path = '/home/kreitnerl/Datasets/syn_ideal/'
    cho = torch.empty(totalEntries).uniform_(0.01,3.5)
    naa = torch.empty(totalEntries).uniform_(0.01,3.5)

    base, _ = os.path.split(path)
    os.makedirs(base,exist_ok=True)
    quantities = {
        'cho':  np.asarray(cho),
        'naa': np.asarray(naa)
    }
    io.savemat(path + 'quantities.mat', do_compression=True, mdict=quantities)
    print('Done. You can find your dataset at', path)

