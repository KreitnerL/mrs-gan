import os.path

# import random
import numpy as np
from pandas import read_csv, DataFrame
from torch.utils.data.sampler import RandomSampler

# import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


# from torch import from_numpy
# from PIL import Image


class AlignedLabeledSpectralDatasetModified(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.root)) # Returns a list of paths of the files in the dataset
        total_count = 0
        file_count = []
        total_paths = []
        for i in range(len(self.AB_paths)):
            with open(self.AB_paths[i],"r") as file:
                AB_spectra = read_csv(file, names=['Spectra','NAA','cho'],
                                      # dtype={'Spectra': np.complexfloating, 'NAA': np.floating, 'cho': np.floating},
                                      low_memory=False)
                # total_count += len(AB_spectra.Spectra)
                # # file_count += [len(AB_spectra.Spectra)]
                # spectra_index = []
                # for _ in range(file_count[i]):
                #     total_paths += [self.AB_paths[i]]
                #     spectra_index += list(range(total_count))

        # Split the data if indicated, save the indices in a CSV file, and set the sampler for the training phase
        # Assumption: if the dataset needs to be split, then chances are that the model has not been trained yet
        if self.opt.split:
            count = list(range(total_count))
            train, val, test = splitData(count, self.opt.val_split, self.opt.test_split)
            # key = {'index': count,
            #        'paths': total_paths}
            # train_paths, val_paths, test_paths = [], [], []
            # for i in range(len(train)):
            #     train_paths += spectra_index[train[i],[1]]
            # for i in range(len(val)):
            #     val_paths += spectra_index[val[i],[1]]
            # for i in range(len(test)):
            #     test_paths += spectra_index[test[i],[1]]    # make sure the
            # contents = {'train': train, 'train_paths': total_paths[train_paths],
            #             'val': val, 'val_paths': total_paths[val_paths],
            #             'test': test, 'test_paths': total_paths[test_paths]}#,
            #             # 'key': key}
            contents = {'train': train,
                        'val': val,
                        'test': test}#,
            indices = DataFrame(contents, columns=['train', 'val', 'test'])
            filename = save_path(self)

            with open(filename, 'w') as file:
                indices.to_csv(file)#, header=True)
            # string = ''
            # with open(string.join([filename,'train.txt']),'w') as file:
            #     file.writelines(str(a))
            # with open(string.join([filename,'validate.txt']),'w') as file:
            #     file.write(str(b))
            # with open(string.join([filename,'test.txt']),'w') as file:
            #     file.write(str(c))

            self.sampler = RandomSampler(train[:])




    def __getitem__(self, index):

        AB_path = self.AB_paths[index]
        with open(AB_path,"r") as file:
            AB_spectra = read_csv(file, names=['Spectra','NAA','cho'],
                                  # dtype={'Spectra': np.complexfloating, 'NAA': np.floating, 'cho': np.floating},
                                  low_memory=False)

        A = AB_spectra.Spectra
        #B = [AB_spectra.NAA, AB_spectra.cho]

        return {'A_data': A, 'A_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'SingleSpectralDataset'

def splitData(count, val_split=0.2, test_split=0.1, *both):
    dataset_size = len(count)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)      # randomize dataset indices
    split = [int(np.floor((1 - val_split - test_split) * dataset_size)), int(np.floor(1 - test_split * dataset_size))]
    train_sampler, valid_sampler, test_sampler = indices[:split[0]], indices[split[0]:split[1]], indices[split[1]:]

    if not both:
        return [RandomSampler(train_sampler)], [RandomSampler(valid_sampler)], [RandomSampler(test_sampler)]
    else:
        return [RandomSampler(train_sampler)], [RandomSampler(valid_sampler)], [RandomSampler(test_sampler)], train_sampler, valid_sampler, test_sampler


def save_path(self):
    save_filename = 'model_phase_indices.csv'
    path = os.path.join(self.opt.save_dir, save_filename)

    return path
