from models.cycleGAN_spectra import CycleGAN_spectra

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
