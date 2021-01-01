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
                                        shuffle=opt.shuffle,   # Already included when the dataset is split
                                        num_workers=int(opt.nThreads),
                                        drop_last=False)

    def createDataset(self, opt, phase):
        dataset = None
        if opt.dataset_mode == 'unaligned':
            from data.unaligned_dataset import UnalignedDataset
            dataset = UnalignedDataset()
        elif opt.dataset_mode == 'dicom_spectral_dataset':
            from data.dicom_spectral_dataset import DicomSpectralDataset
            dataset = DicomSpectralDataset()
        elif opt.dataset_mode == 'spectra_component_dataset':
            from data.spectra_components_dataset import SpectraComponentDataset
            dataset = SpectraComponentDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

        dataset.initialize(opt, phase)
        if not opt.quiet:
            print("dataset [%s] was created" % (dataset.name()))
        return dataset

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), float(self.opt.max_dataset_size))
