###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
from util import util
from pandas import read_csv

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
DATA_EXTENSIONS = [
    'proc.dcm','.mat','.h5'
]

data_is_image=False

def is_data_file(filename, extensions):
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir, opt=None):
    images = []
    global data_is_image
    if os.path.exists(dir):
        assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    else:
        util.mkdir(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if opt is not None and opt.file_ext is not None:
                for i in range(len(opt.file_ext)):
                    if fname.endswith(opt.file_ext[i]):
                        path = os.path.join(root, fname)
                        images.append(path)
            elif is_data_file(fname, DATA_EXTENSIONS):
                path = os.path.join(root, fname)
                images.append(path)
            elif is_data_file(fname, IMG_EXTENSIONS):
                data_is_image = True
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    if data_is_image:
        return Image.open(path).convert('RGB')
    return read_csv(path)


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
