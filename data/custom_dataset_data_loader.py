from torch.utils.data import DataLoader
from data.base_data_loader import BaseDataLoader

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, phase):
        BaseDataLoader.initialize(self, opt)
        self.dataset = self.createDataset(opt, phase)

        self.dataloader = DataLoader(self.dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=(opt.shuffle and phase=='train'),   # Already included when the dataset is split
                                        num_workers=int(opt.nThreads),
                                        drop_last=False)

    def createDataset(self, opt, phase):
        dataset = None
        if opt.dataset_mode == 'spectra_component_dataset':
            from data.spectra_components_dataset import SpectraComponentDataset
            dataset = SpectraComponentDataset()
        elif opt.dataset_mode == 'reg_cyclegan_dataset':
            from data.reg_cyclegan_dataset import RegCycleGANDataset
            dataset = RegCycleGANDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

        dataset.initialize(opt, phase)
        if not opt.quiet:
            print("dataset [%s] was created" % (dataset.name()))
        return dataset

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
