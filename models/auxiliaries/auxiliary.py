import functools
import torch.nn as nn
from torch.nn import init

num_dimensions = 2

def set_num_dimensions(num_dim):
    global num_dimensions
    num_dimensions = num_dim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_weights(net, init_type='normal', init_gain=0.02, activation='leaky_relu'):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=activation)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

conv = {
    1: nn.Conv1d,
    2: nn.Conv2d
}
transpose_conv = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d
}
def get_conv():
    return conv[num_dimensions]
def get_conv_transpose():
    return transpose_conv[num_dimensions]


reflection_padding = {
    1: nn.ReflectionPad1d,
    2: nn.ReflectionPad2d
}
replication_padding = {
    1: nn.ReplicationPad1d,
    2: nn.ReplicationPad2d
}
def get_padding(type: str):
    if type == 'reflect':
        return reflection_padding[num_dimensions]
    elif type == 'replicate':
        return replication_padding[num_dimensions]
    else:
        raise NotImplementedError('padding type [%s] is not found' % type)



batch_norm = {
    1: functools.partial(nn.BatchNorm1d, affine=True),
    2: functools.partial(nn.BatchNorm2d, affine=True)
}
instance_norm = {
    1: functools.partial(nn.InstanceNorm1d, affine=False),
    2: functools.partial(nn.InstanceNorm2d, affine=False)
}
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return batch_norm[num_dimensions]
    elif norm_type == 'instance':
        return instance_norm[num_dimensions]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


avg_pooling = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d
}
max_pooling = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d
}
def get_pooling(pool_type: str):
    if pool_type == 'avg':
        return avg_pooling[num_dimensions]
    elif pool_type == 'max':
        return max_pooling[num_dimensions]
    else:
        raise NotImplementedError('Pooling type [%s] not implemented' % pool_type)

adaptive_avg_pooling = {
    1: nn.AdaptiveAvgPool1d,
    2: nn.AdaptiveAvgPool2d
}
adaptive_max_pooling = {
    1: nn.AdaptiveMaxPool1d,
    2: nn.AdaptiveMaxPool2d
}
def get_adaptive_pooling(pool_type: str):
    if pool_type == 'avg':
        return adaptive_avg_pooling[num_dimensions]
    elif pool_type == 'max':
        return adaptive_max_pooling[num_dimensions]
    else:
        raise NotImplementedError('Pooling type [%s] not implemented' % pool_type)
