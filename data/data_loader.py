
def CreateDataLoader(opt, phase):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, phase)
    return data_loader
