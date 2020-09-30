
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
