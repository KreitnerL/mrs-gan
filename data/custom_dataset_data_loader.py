from torch.utils.data import DataLoader
from data.base_data_loader import BaseDataLoader



def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    # elif opt.dataset_mode == 'LabeledDicomSpectralDataset':
    #     from data.labeled_dicom_spectral_dataset import LabeledDicomSpectralDataset
    #     dataset = LabeledDicomSpectralDataset()
    elif opt.dataset_mode == 'LabeledMatSpectralDataset':
        from data.labeled_mat_spectral_dataset import LabeledMatSpectralDataset
        dataset = LabeledMatSpectralDataset()
    # elif opt.dataset_mode == 'LabeledMatSpectralDataset_Simplified':
    #     from data.labeled_mat_spectral_dataset_simplified import LabeledMatSpectralDataset_Simplified
    #     dataset = LabeledMatSpectralDataset_Simplified()
    # elif opt.dataset_mode == 'preprocessspectra':
    #     from data.preprocess_spectra import PreprocessDicomSpectralDataset
    #     dataset = PreprocessDicomSpectralDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    dataset.initialize(opt)
    print("dataset [%s] was created" % (dataset.name()))
    return dataset


# class CustomDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'

#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)

#         if opt.phase=='train':
#             self.dataloader = DataLoader(self.dataset,
#                                          batch_size=opt.batchSize,
#                                          shuffle=not opt.serial_batches,   # Already included when the dataset is split
#                                          num_workers=int(opt.nThreads),
#                                          drop_last=False)
#         elif opt.phase=='val':
#             self.dataloader = DataLoader(self.dataset,
#                                          batch_size=opt.batchSize,#len(self.dataset),
#                                          shuffle=not opt.serial_batches,   # Already included when the dataset is split
#                                          num_workers=int(opt.nThreads),
#                                          drop_last=False)


#     def load_data(self):
#         return self.dataloader

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)


# class CrossValidationDatasetLoader(BaseDataLoader):
#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.k = opt.k_folds
#         self.opt = opt
#         # opt.add_argument()
#         self.dataset = CreateDataset(opt)
#         self.number = len(self.dataset)

#     def name(self):
#         return '{}-Fold_CrossValidationDatasetLoader'.format(self.k)

#     def partitions(self, number, k):
#         partitions = np.ones(k) * int(number / k)
#         partitions[0:(number % k)] += 1
#         return partitions

#     def get_indices(self, k, number):
#         fold_sizes = self.partitions(number, k)
#         indices = np.arange(number).astype(int)
#         current = 0
#         for fold_size in fold_sizes:
#             start = current
#             stop = current + fold_size
#             current = stop
#             yield(indices[int(start):int(stop)])

#     def k_folds(self):
#         indices = np.arange(self.number)
#         for test_idx in self.get_indices(self.k, self.number):
#             train_idx = np.setdiff1d(indices, test_idx)
#             yield train_idx, test_idx

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)

#     def load_dataset(self):
#         return self.dataset
