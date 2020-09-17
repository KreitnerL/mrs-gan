###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import os
import os.path

import torch.utils.data as data
# from pandas import read_csv

from util import util

# import pydicom
# from pydicom.data import get_testdata_files

IMG_EXTENSIONS = ['proc.dcm','.mat','.h5']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, opt):
    data = []
    count = 0
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    else:
        util.mkdir(dir)

    for root, dirs, fnames in sorted(os.walk(dir)):
        # print('root, dirs, fnames = ',[root, dirs, fnames])
        # if root.endswith(opt.folder_ext):
        for fname in fnames:
            count += 1
            for i in range(len(opt.file_ext)):
                if fname.endswith(opt.file_ext[i]):
                    # print('count = ', count)
                    path = os.path.join(root, fname)
                    data.append(path)

    return data


def default_loader(path):
    return read_csv(path)


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 files in: " + root + "\n"
                               "Supported file extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not False:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
