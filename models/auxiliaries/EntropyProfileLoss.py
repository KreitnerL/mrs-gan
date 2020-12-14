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
    Updated: Nov 19, 2020
    '''
    def __init__(self, kernel_sizes=(2, 3, 4, 5, 6, 7)):
        super(EntropyProfileLoss, self).__init__()
        self.k = tuple( 2**val for val in kernel_sizes )
        self.entropyloss = nn.Sequential()
        for k in self.k:
            self.entropyloss.add_module('kernel{}'.format(k), Loss(k))

    def forward(self, input, target):
        out = torch.tensor(0.0,requires_grad=True, device='cuda')
        _, _, out = self.entropyloss((input, target, out))
        return out

class Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.entropy = Entropy(k)
        self.loss = nn.L1Loss()

    def forward(self, args):
        input, target, out = args[0], args[1], args[2]
        out = out + self.loss(self.entropy(input), self.entropy(target))
        return input, target, out


class Entropy(nn.Module):
    def __init__(self, k):
        super(Entropy, self).__init__()
        self.initialized = False
        self.k = k

    def initialize(self, input: torch.Tensor):
        self.initialized = True
        self.s = input.shape
        self.index = torch.empty(1,1,(self.s[2] - self.k + 1) * self.k, device='cuda', dtype=torch.long)
        for i in range(input.shape[2]-self.k+1):
            m = i*self.k
            self.index[:,:,m:m+self.k] = torch.arange(i,i+self.k,1)
        self.index = self.index.expand(self.s[0],self.s[1],-1)        

    def forward(self, input: torch.Tensor):
        if not self.initialized:
            self.initialize(input)

        temp = input.gather(2, self.index)
        temp = temp.reshape(self.s[0],self.s[1],self.s[2] - self.k + 1,self.k)

        return torch.sum(nn.functional.softmax(temp, dim=3) * nn.functional.log_softmax(temp, dim=3),dim=3)