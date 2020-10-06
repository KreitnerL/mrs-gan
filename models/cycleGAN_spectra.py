from models.cycleGAN import CycleGANModel
import torch
from collections import OrderedDict
import util.util as util
T = torch.Tensor


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

class CycleGAN_spectra(CycleGANModel):
    """
    This class implements a CycleGAN model for learning 1d signal translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def name(self):
        return 'CycleGAN_spectra'

    def __init__(self, opt):
        super().__init__(opt, 1)

    def get_current_visuals(self):
        x = list(range(self.real_A.size()[-1]))
        real_A = util.get_img_from_fig(x, self.real_A.data, 'PPM')
        fake_B = util.get_img_from_fig(x, self.fake_B.data, 'PPM')
        rec_A = util.get_img_from_fig(x, self.rec_A.data, 'PPM')
        real_B = util.get_img_from_fig(x, self.real_B.data, 'PPM')
        fake_A = util.get_img_from_fig(x, self.fake_A.data, 'PPM')
        rec_B = util.get_img_from_fig(x, self.rec_B.data, 'PPM')

        if self.opt.identity > 0.0:
            idt_A = util.get_img_from_fig(x, self.idt_A.data, 'PPM')
            idt_B = util.get_img_from_fig(x, self.idt_B.data, 'PPM')
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
