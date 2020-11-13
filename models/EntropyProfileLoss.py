import torch
import torch.nn as nn

__all__ = ['EntropyProfileLoss']


class EntropyProfileLoss(nn.Module):
    '''
    This function calculates the entropy profile of 1D signals at various scales and
    returns the L1 loss comparing the entropy profiles of the input and target data.

    Args:
        kernel_sizes: tuple     exponents (base 2) indicating the size of the running kernels
        gpu_ids:      list      list of GPUs to make use of nn.DataParallel

    Output:
        out:          tensor    summation of the L1 loss scores for all of the kernels

    Ing. John T LaMaster Nov 14, 2019
    '''
    def __init__(self, kernel_sizes=(2, 3, 4, 5, 6, 7), gpu_ids=[]):
        super(EntropyProfileLoss, self).__init__()
        self.k = tuple( 2**val for val in kernel_sizes )
        if len(gpu_ids) > 0:
            self.entropy = nn.DataParallel(Entropy(),device_ids=gpu_ids)
            self.loss = nn.DataParallel(nn.L1Loss(),device_ids=gpu_ids)
        else:
            self.entropy = Entropy()
            self.loss = nn.L1Loss()

    def forward(self, input, target):
        out = torch.empty(len(self.k), device='cuda')
        for i,k in enumerate(self.k):
            I = self.entropy(k, input)
            T = self.entropy(k, target)
            out[i] = self.loss(I,T)
        out = torch.sum(out)
        return out

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, k, input): # Todo: add 'cpu' functionality
        s = input.size()
        temp = torch.empty(s[0],s[1],s[2] - k + 1, k, device='cuda')
        for i in range(temp.size(2)):
            temp[:,:,i,:] = input[:,:,i:i + k]
        return torch.sum(nn.functional.softmax(temp, dim=3) * nn.functional.log_softmax(temp, dim=3),dim=3)
