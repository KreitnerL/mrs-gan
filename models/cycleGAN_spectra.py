from models.cycleGAN import CycleGANModel
import torch
import numpy as np
from collections import OrderedDict
import util.util as util
T = torch.Tensor


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
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None
        if hasattr(self, 'real_A'):
            x = np.linspace(*self.opt.ppm_range, self.real_A.size()[-1])
            real_A = util.get_img_from_fig(x, self.real_A[0:1].data, 'PPM')
            fake_B = util.get_img_from_fig(x, self.fake_B[0:1].data, 'PPM')
            rec_A = util.get_img_from_fig(x, self.rec_A[0:1].data, 'PPM')
        if hasattr(self, 'real_B'):
            x = list(range(self.real_B.size()[-1]))
            real_B = util.get_img_from_fig(x, self.real_B[0:1].data, 'PPM')
            fake_A = util.get_img_from_fig(x, self.fake_A[0:1].data, 'PPM')
            rec_B = util.get_img_from_fig(x, self.rec_B[0:1].data, 'PPM')

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
