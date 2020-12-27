import torch
import torch.nn as nn
import torchvision
from torch.nn.utils import spectral_norm

import math
import numpy as np

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

class Encoder(nn.Module):
    def __init__(self, input_nc: int, ngf=64, norm_layer=get_norm_layer('batch'), n_downsampling=2, cbam=False):
        """
        Create an Encoder network that downsamples the given input and encodes it into a sparse feature representation.
        Parameters
        ----------
            - input_nc (int)      -- the number of channels in input images
            - ngf (int)           -- the number of filters in the last conv layer
            - norm_layer          -- normalization layer
            - n_downsampling (int)-- the number of downsampling steps
        """ 
        super().__init__()
        model = [get_padding('reflect')(3),
                 get_conv()(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        # if cbam:
        #     model.append(CBAM1d(ngf))

        for i in range(n_downsampling):
            mult = 2**i
            model += [get_conv()(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            # if cbam:
            #     model.append(CBAM1d(ngf * mult * 2))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, norm_layer=get_norm_layer('batch'), n_upsampling=2, cbam=False):
        """
        Create a Decoder network that upsamples the given input and decodes it from a sparse feature representation.
        Parameters
        ----------
            - output_nc (int)     -- the number of channels in output images
            - ngf (int)           -- the number of filters in the last conv layer
            - norm_layer          -- normalization layer
            - n_upsampling (int)-- the number of downsampling steps
        """ 
        super().__init__()
        model = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            model += [get_conv_transpose()(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            # if cbam:
            #     model.append(CBAM1d(ngf * mult * 2))
        model += [get_padding('reflect')(3)]
        model += [get_conv()(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Transformer(nn.Module):
    def __init__(self, input_nc: int, norm_layer=get_norm_layer('batch'), n_blocks=4, padding_type='zero', cbam=False):
        """
        Create a Transformer network that applies style transform on a sparse feature representation.
        Parameters:
        ----------
            - input_nc (int) -- the number of channels in input images
            - ngf (int)      -- the number of filters in the last conv layer
            - norm_layer     -- normalization layer
            - n_blocks (int)      -- the number of ResNet blocks
            - padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()
        model = []
        if cbam:
            model.append(CBAM1d(input_nc))
        for i in range(n_blocks):
            model += [ResnetBlock(input_nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False)]
        if cbam:
            model.append(CBAM1d(input_nc))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ExtractorConv(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction
    """
    def __init__(self, in_out = (1,1), ndf=32, n_layers=3, norm_layer=get_norm_layer('instance'), data_length=1024, gpu_ids=[], cbam=False):
        super(ExtractorConv, self).__init__()
        self.gpu_ids = gpu_ids

        kernel_size=4
        padding=1
        stride=2

        self.sequence = nn.ModuleList([])
        c_in = in_out[0]
        c_out = ndf
        # Scale down tensor of length L to L/(2**n_layers)
        # Simultaniously upscale Feature dimension C to 2**_n_layers 
        for _ in range(n_layers):
            self.sequence.extend([
                get_conv()(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
                norm_layer(c_out),
                nn.LeakyReLU()
            ])
            if cbam:
                self.sequence.extend([CBAM1d(c_out)])
            c_in = c_out
            c_out *= 2

        self.sequence.extend([
            get_conv()(c_in, 1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(int(data_length / (2**(n_layers+1))), in_out[1]),
            nn.Sigmoid()
        ])

    def forward(self, input):
        for layer in self.sequence:
            input = layer(input)
        return input.squeeze()

class ExtractorMLP(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction
    """
    def __init__(self, in_out = (1,1), num_neurons=(100,100,100), norm_layer=get_norm_layer('instance'), gpu_ids=[], cbam=False):
        super(ExtractorMLP, self).__init__()
        self.gpu_ids = gpu_ids

        self.layers = nn.Sequential()
        self.layers.add_module('Flatten', nn.Flatten())
        num_neurons = [in_out[0], *num_neurons, in_out[1]]
        for i in range(1, len(num_neurons)):
            self.layers.add_module('Linear'+str(i), nn.Linear(num_neurons[i-1], num_neurons[i]))
            if i < len(num_neurons)-1:
                self.layers.add_module('LeakyReLU'+str(i), nn.LeakyReLU())
            if cbam:
                self.layers.add_module('Unsqueeze'+str(i), LambdaModule(lambda x: x.unsqueeze(-1)))
                self.layers.add_module('CBAM'+str(i), CBAM1d(num_neurons[i]))
                self.layers.add_module('Squeeze'+str(i), LambdaModule(lambda x: x.squeeze(-1)))
        self.layers.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, input):
        out = self.layers(input)
        return out

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False, n_blocks=4, gpu_ids=[], padding_type='zero'):
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
        assert(n_blocks >= 0)
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
            model += [get_conv()(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

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

# Defines the Unet generator.
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=get_norm_layer('batch'), use_dropout=False, gpu_ids=[]):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=get_norm_layer('batch'), use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = get_conv()(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = get_conv_transpose()(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = get_conv_transpose()(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = get_conv_transpose()(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class SpectraNLayerDiscriminator(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction
    """
    def __init__(self, input_nc, ndf=32, n_layers=3, norm_layer=get_norm_layer('instance'), data_length=1024, gpu_ids=[], cbam=False, output_nc=1):
        super(SpectraNLayerDiscriminator, self).__init__()
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
                get_conv()(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
                norm_layer(c_out),
                nn.LeakyReLU(0.2, True)
            ])
            if cbam:
                self.sequence.extend([CBAM1d(c_out)])
            c_in = c_out
            c_out *= 2

        self.sequence.extend([
            get_conv()(c_in, 1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Flatten(),
            nn.Linear(int(data_length / (2**(n_layers+1))), output_nc)
        ])

    def forward(self, input):
        for layer in self.sequence:
            input = layer(input)
        return input

class SpectraNLayerDiscriminator_SN(nn.Module):
    """
    Defines a Discriminator Network that scales down a given spectra of size L to L/(2*n_layers) with convolution, flattens it
    and finally uses a Linear layer to compute a scalar that represents the networks prediction.
    Additionally, the spectral norm of each layer will be contrained by spectral normalization to control the Lipschitz constant.
    """
    def __init__(self, input_nc, ndf=32, n_layers=3, data_length=1024, gpu_ids=[]):
        super(SpectraNLayerDiscriminator_SN, self).__init__()
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
                spectral_norm(get_conv()(c_in, c_out,
                          kernel_size=kernel_size, stride=stride, padding=padding)),
                nn.LeakyReLU(0.2, True)
            ])
            c_in = c_out
            c_out *= 2

        self.sequence.extend([
            spectral_norm(get_conv()(c_in, 1, kernel_size=kernel_size, stride=stride, padding=padding)),
            nn.Flatten(),
            spectral_norm(nn.Linear(int(data_length / (2**(n_layers+1))), 1))
        ])

    def forward(self, input):
        for layer in self.sequence:
            input = layer(input)
        return input
        

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=get_norm_layer('batch'), gpu_ids=[]):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            get_conv()(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                get_conv()(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            get_conv()(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [get_conv()(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class ResNet34(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_drop = nn.Dropout(p=0.75)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



class FeatureResNet34(nn.Module):
    def __init__(self, gpu_ids, **kwargs):
        super(FeatureResNet34, self).__init__()
        self.gpu_ids = gpu_ids
        self.resnet = ResNet34(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
        self.resnet.load_state_dict(torch.utils.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet34']))
        for param in self.resnet.parameters():
            param.requires_grad = False


    def forward(self, input):
        return self.resnet(input)
