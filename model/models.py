
def create_model(opt, *k):
    model = None
    # print(opt.model)
    if opt.model == 'fitting':
        assert(opt.dataset_mode == 'single' or opt.dataset_mode == 'single_modified' or
               opt.dataset_mode == 'DicomSpectralDataset' or 'SpectralDataset' in opt.dataset_mode)
        from model.FittingModel import FittingModel
        model = FittingModel()
    elif 'forward' in opt.model:
        assert(opt.dataset_mode == 'single' or opt.dataset_mode == 'single_modified' or
               opt.dataset_mode == 'DicomSpectralDataset' or 'SpectralDataset' in opt.dataset_mode)
        from model.ForwardFittingModel import ForwardFittingModel
        model = ForwardFittingModel()
    # elif 'randomforest' in opt.model:
    #     assert(opt.dataset_mode == 'single' or opt.dataset_mode == 'single_modified' or
    #            opt.dataset_mode == 'DicomSpectralDataset' or 'SpectralDataset' in opt.dataset_mode)
    #     from model.RandomForestRegression import

    # elif opt.model == 'test':   # TODO: sort through this later
    #     assert(opt.dataset_mode == 'single' or opt.dataset_mode == 'single_modified')
    #     from .test_model import TestModel
    #     model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.k_folds != -1:
        model.initialize(opt, k)
    else:
        model.initialize(opt)

    print("model [%s] was created" % (model.name()))
    return model


