import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
from models.auxiliary import (convertdict, get_actvn_layer, get_conv_layer,
                              get_norm_layer, get_padding_layer)
from modules.learned_group_modules import *
from numpy import linalg as LA
# from numpy import fft
from torch import fft, rfft

__all__ = ['ResnetEstimator']

##############################################################################
# Classes
##############################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# Fast Neural Style Transfer
# https://github.com/jcjohnson/fast-neural-style/
# SE block code come from moskomule
# https://github.com/moskomule/senet.pytorch/
# Residual Unit Pre-Activation comes from
#   Identity Mappings in Deep Residual Netowrks by Kaiming He 2016
# SELU: Scaled Exponential Linear Unites
#   Self-Normalizing Neural Networks by Klambauer 2017

# netEst = define.ResnetEstimator(dim=1, input_nc=2, output_nc=20,padding_type='reflection', actvn_type='selu', use_dropout=True,n_blocks=3, gpu_ids=gpu_ids, se=True, n_downsampling=2, pAct=True)
class ResnetEstimator(nn.Module):
    def __init__(self, dim, input_nc, output_nc, padding_type, norm_type, depth=1, ngf=64, actvn_type='relu',
                 use_dropout=False, use_sigmoid=True, n_blocks=3, gpu_ids=[], se=False, n_downsampling=2, pAct=False, **kwargs):
        """
        Args:
            dim = dimensionality of the input (for a single channel)
            input_nc = number of input channels
            output_nc = number of output channels
        """
        assert(n_blocks >= 0)
        super(ResnetEstimator, self).__init__()
        model = []
        self.gpu_ids = gpu_ids
        SELU = False if actvn_type != 'selu' else True
        n = int(n_blocks/2)

        model += get_conv_layer(dim=dim, net='conv', in_c=input_nc, out_c=ngf, kernel=4, stride=2, pad=1) # arg = [in_channel, out_channel, kernel_size, stride, padding]
        if not SELU: model += get_norm_layer(dim, norm_type, arg=ngf)
        model += get_actvn_layer(actvn_type)
        # print('model.type(): ',len(model))


        #  Residual Encoding of the Spectra
        mult = 0
        for i in range(n):
            mult = 2**i
            for _ in range(depth):
                model += [ResnetBlock(dim=dim, input_nc=ngf * mult, output_nc=ngf * mult, padding_type=padding_type,
                                      norm_type=norm_type, actvn_type=actvn_type, use_dropout=use_dropout, se=se,
                                      SELU=SELU, pAct=pAct)]
                if not pAct:
                    if not SELU: model += get_norm_layer(dim, norm_type, arg=ngf)
                    model += get_actvn_layer(actvn_type)

            # Transition layer - convolutional downsampling
            if i < n-1: in_c, out_c = ngf * mult, ngf * mult * 2
            else: in_c, out_c = ngf * mult, ngf * mult / 2

            model += get_conv_layer(dim=dim, net='conv', in_c=in_c, out_c=out_c, kernel=4, stride=2, pad=1)

        for i in range(n):
            if i < n-1:
                mult = 2**(n-(i+2))
                in_c, out_c = ngf * mult, ngf * mult
            else: in_c, out_c = output_nc, output_nc
            for _ in range(depth):
                model += [ResnetBlock(dim=dim, input_nc=in_c, output_nc=out_c, padding_type=padding_type,
                                      norm_type=norm_type, actvn_type=actvn_type, use_dropout=use_dropout, se=se,
                                      SELU=SELU, pAct=pAct)]
                if not pAct:
                    if not SELU: model += get_norm_layer(dim, norm_type, arg=out_c)
                    model += get_actvn_layer(actvn_type)

            # Transition layer - convolutional downsampling
            if i < n-2: model += get_conv_layer(dim=dim, net='conv', in_c=in_c, out_c=out_c / 2, kernel=4, stride=2, pad=1)
            elif i==n-2: model += get_conv_layer(dim=dim, net='conv', in_c=in_c, out_c=output_nc, kernel=4, stride=2, pad=1)

        model += get_conv_layer(dim=dim, net='conv', in_c=output_nc, out_c=output_nc, kernel=4, stride=1, pad=0) # arg = [in_channel, out_channel, kernel_size, stride, padding]

        if use_sigmoid: model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)


    def forward(self, input):
        if len(self.gpu_ids)>0 and isinstance(input.data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            out = self.model(input)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, input_nc, output_nc, padding_type, norm_type, actvn_type, use_dropout, se, SELU, pAct):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, input_nc, output_nc, padding_type, norm_type, actvn_type, use_dropout, se, SELU, pAct)
        self.sel = se

    def build_conv_block(self, dim, input_nc, output_nc, padding_type, norm_type, actvn_type, use_dropout, se, SELU, pAct):
        conv_block = []

        if pAct:        # Residual Unit Pre-Activation: Identity mappings in Deep Residual Networks Kamming He, 2016
            if not SELU: conv_block += get_norm_layer(dim, norm_type, input_nc)
            conv_block += get_actvn_layer(actvn_type)

        p = 0
        if padding_type=='zero':
            p = 1
        elif not padding_type=='zero':
            conv_block += get_padding_layer(dim, padding_type)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += get_conv_layer(dim, net='conv', in_c=input_nc, out_c=output_nc, kernel=3, pad=p)
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        if not SELU: conv_block += get_norm_layer(dim, norm_type, output_nc)
        conv_block += get_actvn_layer(actvn_type)

        if not padding_type=='zero':
            conv_block += get_padding_layer(dim, padding_type)

        conv_block += get_conv_layer(dim, net='conv', in_c=output_nc, out_c=output_nc, kernel=3, pad=p)
        if use_dropout: conv_block += [nn.Dropout(0.5)]

        # If original resblock: aka not pAct
        if not pAct:
            if not SELU: conv_block += get_norm_layer(dim, norm_type, output_nc)
            conv_block += get_actvn_layer(actvn_type)

        if se:
            self.squeeze_and_excite = SELayer(dim, input_nc)

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        if self.sel:
            out = self.squeeze_and_excite(out)

        return x + out


class SELayer(nn.Module):
    def __init__(self, dim, channel, reduction=16):
        super(SELayer, self).__init__()
        assert (0<dim and dim<=3), 'SELayer dim must be between 1 and 3'
        squeeze = [nn.AdaptiveAvgPool1d(1)]
        excite = [nn.Linear(channel, channel // reduction, bias=False),
                  nn.ReLU(inplace=True),
                  nn.Linear(channel // reduction, channel, bias=False),
                  nn.Sigmoid()]
        self.squeeze = nn.Sequential(*squeeze)
        self.excite = nn.Sequential(*excite)
        self.case = dim

    def forward(self, x):
        if self.case==1:
            b, c, _ = x.size()                          # 3D tensor for 1D signal (spectra)
            out = self.squeeze(x).view(b,c)
            out = self.excite(out).view(b,c,1)
        elif self.case==2:
            b, c, _, _ = x.size()                     # 4D tensor for 2D signal (images)
            out = self.squeeze(x).view(b,c)
            out = self.excite(out).view(b,c,1,1)
        else:
            b, c, _, _, _ = x.size()                     # 4D tensor for 2D signal (images)
            out = self.squeeze(x).view(b,c)
            out = self.excite(out).view(b,c,1,1,1)

        return x * out.expand_as(x)
