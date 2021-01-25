import torch
import torch.nn as nn

from models.auxiliaries.auxiliary import *
from models.auxiliaries.CBAM import CBAM1d


##############################################################################
# Classes
##############################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wasserstein.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wasserstein':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented!', gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, input, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode == 'wasserstein':
            if target_is_real:
                loss = input.mean()
            else:
                loss = -input.mean()
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = self.loss(input, target_tensor)
        return loss

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ExtractorMLP(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction
    """
    def __init__(self, in_out = (1,1), num_neurons=(100,100,100), norm_layer=get_norm_layer('instance'), gpu_ids=[]):
        super(ExtractorMLP, self).__init__()
        self.gpu_ids = gpu_ids

        self.layers = nn.Sequential()
        self.layers.add_module('Flatten', nn.Flatten())
        num_neurons = [in_out[0], *num_neurons, in_out[1]]
        for i in range(1, len(num_neurons)):
            self.layers.add_module('Linear'+str(i), nn.Linear(num_neurons[i-1], num_neurons[i]))
            if i < len(num_neurons)-1:
                self.layers.add_module('LeakyReLU'+str(i), nn.LeakyReLU())
        self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, input):
        out = self.layers(input)
        return out

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False, n_blocks=4, gpu_ids=[], padding_type='zero', cbam=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [get_padding('reflect')(3),
                get_conv()(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [get_conv()(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, cbam=cbam)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [get_conv_transpose()(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [get_padding('reflect')(3)]
        model += [get_conv()(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, cbam=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, cbam=cbam)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, cbam=False):
        conv_block = []
        p = 0
        if padding_type == 'zero':
            p = 1
        else:
            conv_block += [get_padding('reflect')(padding_type)]

        conv_block += [get_conv()(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'zero':
            p = 1
        else:
            conv_block += [get_padding('reflect')(padding_type)]

        conv_block += [get_conv()(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        if cbam:
            conv_block.append(CBAM1d(dim))

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class NLayerDiscriminator(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction
    """
    def __init__(self, input_nc, ndf=32, n_layers=3, norm_layer=get_norm_layer('instance'), data_length=1024, gpu_ids=[], cbam=False, output_nc=1):
        super(NLayerDiscriminator, self).__init__()
        assert data_length%(2**(n_layers+1))==0
        self.gpu_ids = gpu_ids

        kernel_size=4
        padding=1
        stride=2

        self.sequence = nn.ModuleList([])
        c_in = input_nc
        c_out = ndf
        # Scale down tensor of length L to L/(2**n_layers)
        # Simultaniously upscale Feature dimension C to 2**_n_layers 
        for _ in range(n_layers):
            self.sequence.extend([
                weight_norm(get_conv()(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)),
                norm_layer(c_out),
                nn.LeakyReLU(0.2, True)
            ])
            if cbam:
                self.sequence.extend([CBAM1d(c_out)])
            c_in = c_out
            c_out *= 2

        self.sequence.extend([
            weight_norm(get_conv()(c_in, 1, kernel_size=kernel_size, stride=stride, padding=padding)),
            nn.Flatten(),
            nn.Linear(int(data_length / (2**(n_layers+1))), output_nc)
        ])

    def forward(self, input):
        for layer in self.sequence:
            input = layer(input)
        return input
