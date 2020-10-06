
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel(opt)
    elif opt.model == 'cycle_gan_spectra':
        assert(opt.dataset_mode == 'dicom_spectral_dataset')
        from .cycleGAN_spectra import CycleGANModel
        model = CycleGANModel(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
