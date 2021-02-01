#https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/#:~:text=Spatial%20attention%20represents%20the%20attention,features%20that%20define%20that%20bird.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# from modules.aux.auxiliary import get_adaptive_pooling_layer, get_conv_layer, get_norm_layer
from models.auxiliaries.auxiliary import get_conv, get_adaptive_pooling


__all__ = ['citation', 'ChannelGate', 'SpatialGate', 'CBAM1d', 'CBAM2d', 'CBAM3d']


citation = OrderedDict({'Title': 'CBAM: Convolutional Block Attention Module',
                        'Authors': 'Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon',
                        'Year': '2018',
                        'Journal': 'ECCV',
                        'Institution': 'Korea Advanced Institute of Science and Technology, Lunit Inc., and Adobe Research',
                        'URL': 'https://arxiv.org/pdf/1807.06521.pdf',
                        'Notes': 'Added the possiblity to switch from SE to eCA in the ChannelGate and updated deprecated sigmoid',
                        'Source Code': 'Modified from: https://github.com/Jongchan/attention-module'})


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, dim=1):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = get_conv()(in_planes, out_planes, kernel_size, groups=groups, stride=stride, padding=padding, bias=bias)
        self.bn = nn.InstanceNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class EfficientChannelAttention(nn.Module):
    def __init__(self, num_channels, gamma=2, b=1, dim=1):
        super(EfficientChannelAttention, self).__init__()
        t =int(torch.abs((torch.log2(torch.tensor(num_channels, dtype=torch.float64)) + b) / gamma))
        k = t if t % 2 else t + 1

        self.conv = get_conv()(1, 1, kernel_size=k, stride=1, padding=int(k/2), bias=False)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = x.transpose(-1,-2)
        out = self.conv(out)
        # out = self.sigmoid(out)
        out = out.transpose(-1, -2)
        return out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], method='efficient', dim=1):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.avg_pool = get_adaptive_pooling('avg')(1)
        self.max_pool = get_adaptive_pooling('max')(1)
        self.sigmoid = nn.Sigmoid()
        if method=='efficient':
            self.attention = EfficientChannelAttention(gate_channels, dim=dim)
        elif method=='mlp':
            self.attention = nn.Sequential(
                nn.Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.attention(avg_pool)
            elif pool_type=='max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.attention(max_pool)
            else:
                raise ValueError(x)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        channel_att_sum = self.sigmoid(channel_att_sum)
        scale = channel_att_sum.expand_as(x) if channel_att_sum.dim()>=3 else channel_att_sum.unsqueeze(-1).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, dim):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False, dim=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM1d(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, no_channel=False):
        super(CBAM1d, self).__init__()
        if no_channel:
            self.ChannelGate = nn.Identity()
        else:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, dim=1)
        # self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, dim=1)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=1)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM2d(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM2d, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, dim=2)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=2)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CBAM3d(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3d, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, dim=3)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(dim=3)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out