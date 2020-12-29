import torch
import torch.nn as nn

__all__ = ['FeatureProfileLoss']


class FeatureProfileLoss(nn.Module):
    '''
    This function calculates the entropy and the conent profile of 1D signals at various scales and
    returns the L1 losses comparing the entropy profiles of the input and target data.

    Args:
    ----
        - kernel_sizes (tuple): exponents (base 2) indicating the size of the running kernels

    Output:
    ------
        - entropy_loss (torch.Tensor): summation of the entropy L1 loss scores for all of the kernels
        - content_loss (torch.Tensor): summation of the content L1 loss scores for all of the kernels

    Ing. John T LaMaster Nov 14, 2019
    Linus Kreitner, Dec 22, 2020
    Updated: Dec 22, 2020
    '''
    def __init__(self, kernel_sizes: tuple=(2, 3, 4, 5, 6, 7)):
        super(FeatureProfileLoss, self).__init__()
        self.k = tuple( 2**val for val in kernel_sizes )
        self.entropyloss = nn.Sequential()
        self.entropy_loss = torch.tensor(0.0,requires_grad=True, device='cuda')
        self.content_loss = torch.tensor(0.0,requires_grad=True, device='cuda')
        for k in self.k:
            self.entropyloss.add_module('kernel{}'.format(k), Loss(k))

    def forward(self, input, target):
        _, _, entropy_loss, content_loss = self.entropyloss((input, target, self.entropy_loss, self.content_loss))
        return 100*entropy_loss, content_loss

class Loss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.entropy = Feature(k)
        self.loss_entropy = nn.L1Loss()
        self.loss_content = nn.MSELoss()

    def forward(self, args):
        input, target, entropy_loss, content_loss = args[0], args[1], args[2], args[3]
        entropy_input, content_input = self.entropy(input)
        entropy_target, content_target = self.entropy(target)
        entropy_loss = entropy_loss + self.loss_entropy(entropy_input, entropy_target)
        content_loss = content_loss + self.loss_content(content_input, content_target)
        return input, target, entropy_loss, content_loss


class Feature(nn.Module):
    def __init__(self, k):
        super(Feature, self).__init__()
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
        """
        Parameters:
        ----------
            - input (torch.Tensor): 1D signal of the shape (B x C x L)  

        Returns
        --------
            - entropy of the signal per kernel. (B x C x L-k+1)
            - sum of signal per kernel. (B x C x L-k+1)
        """
        if not self.initialized:
            self.initialize(input)

        temp = input.gather(2, self.index)
        temp = temp.reshape(self.s[0],self.s[1],self.s[2] - self.k + 1,self.k)

        entropy = torch.sum(nn.functional.softmax(temp, dim=3) * nn.functional.log_softmax(temp, dim=3),dim=3)
        content = torch.sum(temp, dim=3)

        return  entropy, content