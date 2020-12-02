import functools
import torch.nn as nn
import torch.nn.functional as F

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
    1: F.avg_pool1d,
    2: F.avg_pool2d
}
max_pooling = {
    1: F.max_pool1d,
    2: F.max_pool2d
}
def get_pooling(pool_type: str):
    if pool_type == 'avg':
        return avg_pooling[num_dimensions]
    elif pool_type == 'max':
        return max_pooling[num_dimensions]
    else:
        raise NotImplementedError('Pooling type [%s] not implemented' % pool_type)

adaptive_avg_pooling = {
    1: F.adaptive_avg_pool1d,
    2: F.adaptive_avg_pool2d
}
adaptive_max_pooling = {
    1: F.adaptive_max_pool1d,
    2: F.adaptive_max_pool2d
}
def get_adaptive_pooling(pool_type: str):
    if pool_type == 'avg':
        return adaptive_avg_pooling[num_dimensions]
    elif pool_type == 'max':
        return adaptive_max_pooling[num_dimensions]
    else:
        raise NotImplementedError('Pooling type [%s] not implemented' % pool_type)
