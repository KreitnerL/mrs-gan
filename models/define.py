from models.networks import *
from models.auxiliaries.auxiliary import init_weights

##############################################################################
# Generator / Discriminator
##############################################################################

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='instance', gpu_ids=[], init_type='normal', cbam=False):
    """Create a generator
    Parameters:
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_32] (for 1x1024 input signals) and [unet_64] (for 1x2048 input signals)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks), [resnet_4blocks] (with 4 Resnet blocks) and [resnet_3blocks] (with 3 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=which_model_netG, gpu_ids=gpu_ids, cbam=cbam)
    if len(gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type, activation='relu')
    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netG, device_ids=gpu_ids)
    else:
        return netG

def define_D(opt, input_nc, ndf, n_layers_D=3, norm='instance', gpu_ids=[], init_type='normal', cbam=False, output_nc=1):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, data_length=opt.data_length, gpu_ids=gpu_ids, cbam=cbam, output_nc=output_nc)   
    if use_gpu:
        netD.cuda()
    init_weights(netD, init_type)

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netD, device_ids=gpu_ids)
    else:
        return netD

def define_extractor(input_nc, output_nc, data_length, ndf, n_layers_D=3, norm='instance', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    # netExtractor = ExtractorConv((input_nc, output_nc), ndf, n_layers=n_layers_D, norm_layer=norm_layer, data_length=opt.data_length, gpu_ids=gpu_ids, cbam=cbam)
    netExtractor = ExtractorMLP((input_nc * data_length, output_nc), num_neurons=[ndf]*n_layers_D, norm_layer=norm_layer, gpu_ids=gpu_ids)
    if use_gpu:
        netExtractor.cuda()
    init_weights(netExtractor, "kaiming", activation='leaky_relu')

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netExtractor, device_ids=gpu_ids)
    else:
        return netExtractor

def define_splitter(input_nc: int, input_length: int, n_p: int, R_num_filter: int, S_num_filter: int, R_num_layers=3, S_num_layers=3, norm='instance', gpu_ids=[]):
    """
    Creates a Splitter network consisting of a style extractor S and a parameter regression network R.
    The given input is individually fed into the style extractor and the parameter regressor.
    
    Parameters:
    -----------  
        - input_dim: tuple(int): Dimensions of the input
        - n_p: int: Number of parameters to predict
        - s_nc: int: Number of style channels
        - R_num_filter: int: Number of filters for the regressor network
        - S_num_filter: int: Number of filters for the style extraction network
        - R_num_layers: int: Number of layers for the regressor network. Default = 3
        - S_num_layers: int: Number of layers for the style extraction network. Default = 3
        - norm: string: Normalization technique. Any of ['instance', 'batch', 'group']. Default = 'instance'
        - gpu_ids: [int]: GPU ids available to this network. Default = []
    """
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    splitter_network = SplitterNetwork(input_nc, input_length, n_p, R_num_filter, S_num_filter, R_num_layers, S_num_layers, get_norm_layer(norm), gpu_ids)
    init_weights(splitter_network, "kaiming", activation='relu')
    if len(gpu_ids):
        return nn.DataParallel(splitter_network, device_ids=gpu_ids)
    else:
        return splitter_network

def define_styleGenerator(content_nc: int, style_nc: int, n_c: int, n_blocks=4, norm='instance', use_dropout=False, padding_type='zero', cbam=False, gpu_ids=[]):
    """
    This ResNet applies the encoded style from the style tensor onto the given content tensor.

    Parameters:
    ----------
        - content_nc (int): number of channels in the content tensor
        - style_nc (int): number of channels in the style tensor
        - n_c (int): number of channels used inside the network
        - n_blocks (int): number of Resnet blocks
        - norm_layer: normalization layer
        - use_dropout: (boolean): if use dropout layers
        - padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero
        - cbam (boolean): If true, use the Convolution Block Attention Module
        - gpu_ids: [int]: GPU ids available to this network. Default = []
    """
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    styleGenerator = StyleGenerator(content_nc, style_nc, n_c, n_blocks=n_blocks, norm_layer=norm_layer, use_dropout=use_dropout, padding_type=padding_type, cbam=False)
    init_weights(styleGenerator, "kaiming", activation='leaky_relu')
    if len(gpu_ids):
        return nn.DataParallel(styleGenerator, device_ids=gpu_ids)
    else:
        return styleGenerator

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
