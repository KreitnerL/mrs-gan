
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycleGAN_PFL':
        assert(opt.dataset_mode == 'unaligned')
        from .cycleGAN_PFL import CycleGAN_PFL
        model = CycleGAN_PFL(opt)
    elif opt.model == 'cycleGAN_spectra':
        assert(opt.dataset_mode == 'dicom_spectral_dataset')
        from .cycleGAN_spectra import CycleGAN_spectra
        model = CycleGAN_spectra(opt)
    elif opt.model == 'cycleGAN_WGP':
        assert(opt.dataset_mode == 'dicom_spectral_dataset')
        from .cycleGAN_WGP import CycleGAN_WGP
        model = CycleGAN_WGP(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
