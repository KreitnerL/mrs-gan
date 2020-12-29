from models.networks import *
from models.auxiliaries.auxiliary import init_weights

##############################################################################
# Generator / Discriminator
##############################################################################

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='instance', gpu_ids=[], init_type='normal'):
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
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=which_model_netG, gpu_ids=gpu_ids)
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

def define_extractor(input_nc, output_nc, data_length, ndf, n_layers_D=3, norm='instance', gpu_ids=[], cbam=False):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    # netExtractor = ExtractorConv((input_nc, output_nc), ndf, n_layers=n_layers_D, norm_layer=norm_layer, data_length=opt.data_length, gpu_ids=gpu_ids, cbam=cbam)
    netExtractor = ExtractorMLP((input_nc * data_length, output_nc), num_neurons=[ndf]*n_layers_D, norm_layer=norm_layer, gpu_ids=gpu_ids, cbam=cbam)
    if use_gpu:
        netExtractor.cuda()
    init_weights(netExtractor, "kaiming", activation='leaky_relu')

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netExtractor, device_ids=gpu_ids)
    else:
        return netExtractor

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
