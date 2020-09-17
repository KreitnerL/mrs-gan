
def CreateDataLoader(opt):
    if opt.k_folds == -1:
        from data.custom_dataset_data_loader import CustomDatasetDataLoader
        data_loader = CustomDatasetDataLoader()
        print(data_loader.name())
        data_loader.initialize(opt)
        return data_loader
    elif opt.k_folds == 0:
        print('Error: opt.k_folds cannot be 0. It must be -1 to not be called or greater than 0.')
    else:#if opt.k_folds >> 0:
        from data.custom_dataset_data_loader import CrossValidationDatasetLoader
        data_loader = CrossValidationDatasetLoader()
        data_loader.initialize(opt)
        print(data_loader.name())
        return data_loader
    # else:

