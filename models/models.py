
def create_model(opt, physicsModel=None):
    model = None
    if opt.model == 'cycleGAN_WGP':
        assert(opt.dataset_mode == 'dicom_spectral_dataset')
        from .cycleGAN_WGP import CycleGAN_WGP
        model = CycleGAN_WGP(opt)
    elif opt.model == 'cycleGAN_WGP_REG':
        assert(opt.dataset_mode == 'spectra_component_dataset')
        from .cycleGAN_WGP_REG import cycleGAN_WGP_REG
        model = cycleGAN_WGP_REG(opt, physicsModel)
    elif opt.model == 'cycleGAN_WSN':
        assert(opt.dataset_mode == 'dicom_spectral_dataset')
        from .cycleGAN_WSN import CycleGAN_WSN
        model = CycleGAN_WSN(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    if not opt.quiet:
        print("model [%s] was created" % (model.name()))
    return model
