# This file lists all the different architecture implementation for a model
# To try other architectures simply call define with another option.

# from modules.loss_networks import *
# from modules.autoencoder_networks import *
# from modules.networks import *
# from modules.spp_resnet import *
import torch
from model.aux.auxiliary import weights_init
from modules.densenet import *
from modules.learned_group_modules import stretch_model
from modules.resnet import *
from modules.vgg import *

# def weights_init(m)
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


__all__ = ['define']


class define():
    # # Define the Generator
    # @staticmethod
    # def Encoder(dim, input_nc, output_nc, ngf, n_blocks=4, norm='batch', use_dropout=False, gpu_ids=[], actvn='selu',
    #             padding='reflect', se=False):
    #     netEn = None
    #     if len(gpu_ids) > 0:
    #         assert(torch.cuda.is_available())
    #
    #     netEn = EncoderGenerator(input_nc=input_nc, output_nc=output_nc, dim=dim, ngf=ngf, norm_type=norm, actvn_type=actvn,
    #                              use_dropout=use_dropout, n_blocks=n_blocks, gpu_ids=gpu_ids, padding_type=padding, se=se)
    #     # else:
    #     #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    #     if len(gpu_ids) > 0:
    #         netEn.cuda()
    #     setattr(netEn, 'GAN_Function','Autoencoder_encoder')
    #     netEn.apply(weights_init)
    #     return netEn
    #
    # # Define the Discriminator
    # @staticmethod
    # def Decoder(dim, input_nc, output_nc, ngf, n_blocks=4, norm='batch', use_dropout=False, gpu_ids=[], actvn='selu',
    #             padding='reflect', se=False):
    #     netDe = None
    #     if len(gpu_ids) > 0:
    #         assert(torch.cuda.is_available())
    #
    #     netDe = DecoderGenerator(input_nc=input_nc, output_nc=output_nc, dim=dim, ngf=ngf, norm_type=norm, actvn_type=actvn,
    #                              use_dropout=use_dropout, n_blocks=n_blocks, gpu_ids=gpu_ids, padding_type=padding, se=se)
    #     if len(gpu_ids) > 0:
    #         netDe.cuda()
    #     setattr(netDe, 'GAN_Function','Autoencoder_decoder')
    #     netDe.apply(weights_init)
    #     return netDe

# netEst = define.Estimator(dim=1, input_nc=2, output_nc=20,padding='reflection', actvn='selu', use_dropout=True,n_blocks=3, gpu_ids=gpu_ids, se=True, n_downsampling=2, pAct=True)
    @staticmethod
    def Estimator(dim, input_nc, output_nc, which_model_netEst, se, use_sigmoid, flexible=False, gpu_ids=[], **kwargs):
        # ngf, n_blocks, n_downsampling, norm_type, actvn_type, padding_type, use_dropout, pAct, depth, learned_g (bool) dense=> num_downs, growth_rate
        netEst = None
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())

        # if which_model_netEst == 'resnet':
        #     netEst = ResnetEstimator(dim=dim, input_nc=input_nc, output_nc=output_nc, se=se,
        #                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        if which_model_netEst == 'resnet50':
            netEst = resnet50(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'resnet18':
            netEst = resnet18(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'resnet34':
            netEst = resnet34(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'resnet101':
            netEst = resnet101(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'resnext50':
            netEst = resnext50_32x4d(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'wideresnet':
            netEst = wide_resnet50_2(input_channels=input_nc, num_classes=output_nc, gpu_ids=gpu_ids)
        elif which_model_netEst == 'vgg19':
            netEst = vgg19_bn(input_channels=input_nc, num_classes=output_nc)
        elif which_model_netEst == 'densenet169':
            netEst = densenet169(input_channels=input_nc, num_classes=output_nc)
        # elif which_model_netEst == 'lgresnet':
        #     netEst = LGResnetEstimator(dim=dim, input_nc=input_nc, output_nc=output_nc, se=se,
        #                                use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'dense': # num_downs=5, growth_rate=2, n_blocks=2
        #     netEst = DenseEstimator(dim=dim, input_nc=input_nc, output_nc=output_nc, se=se,
        #                             use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'resnetspp18':
        #     netEst = resnetspp18(input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'resnetspp34':
        #     netEst = resnetspp34(input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'resnetspp50':
        #     netEst = resnetspp50(input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'resnetspp101':
        #     netEst = resnetspp101(input_nc=input_nc, num_classes=output_nc, se=se,
        #                           use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'resnetspp152':
        #     netEst = resnetspp152(input_nc=input_nc, num_classes=output_nc, se=se,
        #                           use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'forwardspp18':
        #     netEst = resnetspp18(forward=True, input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'forwardspp34':
        #     netEst = resnetspp34(forward=True, input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'forwardspp50':
        #     netEst = resnetspp50(forward=True, input_nc=input_nc, num_classes=output_nc, se=se,
        #                          use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'forwardspp101':
        #     netEst = resnetspp101(forward=True, input_nc=input_nc, num_classes=output_nc, se=se,
        #                           use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)
        # elif which_model_netEst == 'forwardspp152':
        #     netEst = resnetspp152(forward=True, input_nc=input_nc, num_classes=output_nc, se=se,
        #                           use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, **kwargs)

        setattr(netEst, 'Function', 'Parameter_Estimator')
        if flexible:
            netEst = stretch_model(netEst,output_nc)
        # netEst.apply(weights_init)

        netEst.apply(weights_init)
        if len(gpu_ids) > 0:
            netEst.cuda()
        return netEst
