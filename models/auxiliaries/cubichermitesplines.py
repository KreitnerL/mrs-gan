'''
https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch

By: chausies

Converted to 3D tensors for PyTorch NN compatability by Ing. John T LaMaster
Nov 13, 202
'''
# import matplotlib.pylab as P
import numpy as np
import torch

__all__ = ['CubicHermiteSplines']


class CubicHermiteSplines():
    def __init__(self, xaxis, signal):
        self.device = signal.device
        self.x = xaxis.expand_as(signal).to(self.device).float()
        self.y = signal.float()
        self.m = (signal[:,:,1:] - signal[:,:,:-1]) / (self.x[:,:,1:] - self.x[:,:,:-1])
        # Now m is [batchSize, 1, length] tensor
        self.m = torch.cat([self.m[:,:,0].unsqueeze(-1), (self.m[:,:,1:] + self.m[:,:,:-1]) / 2, self.m[:,:,-1].unsqueeze(-1)], dim=-1)

    @staticmethod
    def h_poly_helper(tt):
        out = torch.empty_like(tt)
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        for r in range(4):
            out[:,:,r,:] = A[r,0] * tt[:,:,0,:] + A[r,1] * tt[:,:,1,:] + \
                           A[r,2] * tt[:,:,2,:] + A[r,3] * tt[:,:,3,:]
        return out

    def h_poly(self, t):
        tt = torch.empty(t.shape[0], t.shape[1], 4, t.shape[2]).to(self.device)
        tt[:,:,0,:].fill_(1.)
        for i in range(1, 4):
            tt[:,:,i,:] = tt[:,:,i-1,:] * t
        return self.h_poly_helper(tt)

    def interp(self, xs):
        I = torch.from_numpy(np.apply_along_axis(np.searchsorted, 2, self.x[:,:,1:].clone().cpu(), xs.cpu())).long().to(self.y.device)# self.x[:,:,1:]
        x = torch.gather(self.x, -1, I)
        dx = torch.gather(self.x, -1, I+1) - x

        hh = self.h_poly((xs - x)/dx)

        return hh[:,:,0,:]*torch.gather(self.y,-1,I)   + hh[:,:,1,:]*torch.gather(self.m,-1,I)*dx   + \
               hh[:,:,2,:]*torch.gather(self.y,-1,I+1) + hh[:,:,3,:]*torch.gather(self.m,-1,I+1)*dx
