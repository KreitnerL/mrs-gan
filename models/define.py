from models.networks import *
from models.auxiliaries.auxiliary import init_weights

##############################################################################
# Generator / Discriminator
##############################################################################

def define_modular_G(input_nc: int, output_nc: int, ngf: int, num_res_blocks: int, norm='instance', use_dropout=False, gpu_ids=[], n_downsampling=2, cbam=False, init_type='normal'):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    netG = Encoder_Transform_Decoder(input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=num_res_blocks, n_downsampling=n_downsampling, cbam=cbam)

    if use_gpu:
        assert(torch.cuda.is_available())

    if len(gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type, activation='relu')
    return netG
    

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='instance', use_dropout=False, gpu_ids=[], init_type='normal'):
    """Create a generator
    Parameters:
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
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
    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=which_model_netG, gpu_ids=gpu_ids)

    # if which_model_netG == 'resnet_12blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_4blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_3blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_32':
    #     netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_64':
    #     netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type, activation='relu')
    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netG, device_ids=gpu_ids)
    else:
        return netG

def define_D(opt, input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='instance', gpu_ids=[], init_type='normal', cbam=False, output_nc=1):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, gpu_ids=gpu_ids)
    elif which_model_netD == 'spectra':
        netD = SpectraNLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, data_length=opt.data_length, gpu_ids=gpu_ids, cbam=cbam, output_nc=output_nc)   
    elif which_model_netD == 'spectra_sn':
        netD = SpectraNLayerDiscriminator_SN(input_nc, ndf, n_layers=n_layers_D, data_length=opt.data_length, gpu_ids=gpu_ids, output_nc=output_nc)  
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda()
    if not which_model_netD == 'spectra_sn':
        init_weights(netD, init_type)

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netD, device_ids=gpu_ids)
    else:
        return netD

def define_extractor(opt, input_nc, ndf, n_layers_D=3, norm='instance', gpu_ids=[], init_type='normal', cbam=False, output_nc=1):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    netExtractor = Extractor(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, data_length=opt.data_length, gpu_ids=gpu_ids, cbam=cbam, output_nc=output_nc)

    if use_gpu:
        netExtractor.cuda()
    init_weights(netExtractor, "kaiming", activation='relu')

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netExtractor, device_ids=gpu_ids)
    else:
        return netExtractor

def define_feature_network(which_model_netFeat, gpu_ids=[]):
    netFeat = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netFeat == 'resnet34':
        netFeat = FeatureResNet34(gpu_ids=gpu_ids)
    # elif which_model_netFeat == 'resnet101':
    #     netFeat = FeatureResNet101(gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Feature model name [%s] is not recognized' %
                                  which_model_netFeat)
    if use_gpu:
        netFeat.cuda()

    if len(gpu_ids): # and isinstance(input.data, torch.cuda.FloatTensor):
        return nn.DataParallel(netFeat, device_ids=gpu_ids)
    else:
        return netFeat


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
