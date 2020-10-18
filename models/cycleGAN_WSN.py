from models.cycleGAN_spectra import CycleGAN_spectra
from collections import OrderedDict

class CycleGAN_WSN(CycleGAN_spectra):
    """
    This class implements a CycleGAN model for learning 1d signal translation without paired data,
    using the wasserstein loss function with spectral normalization.
    The model training requires '--dicom_spectral_dataset' dataset.
    """

    def name(self):
        return 'CycleGAN_WGP'

    def __init__(self, opt):
        opt.gan_mode = 'wgangp'
        opt.which_model_netD = 'spectra_sn'
        super().__init__(opt)

    def get_current_losses(self):
        D_A = self.loss_D_A.data
        G_A = -self.loss_G_A.data
        Cyc_A = self.loss_cycle_A.data
        D_B = self.loss_D_B.data
        G_B = -self.loss_G_B.data
        Cyc_B = self.loss_cycle_B.data
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data
            idt_B = self.loss_idt_B.data
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])
