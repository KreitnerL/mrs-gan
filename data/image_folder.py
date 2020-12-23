import os
import os.path
from util import util

def get_file_extensions(file_type):
    if file_type == 'numpy':
        return ['.npz']
    elif file_type == 'image':
        return ['.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    elif file_type == 'dicom':
        return ['.dcm']
    elif file_type == 'matlab':
        return ['.mat']
    else:
        raise ValueError("file type [%s] not recognized." % file_type)

def is_data_file(filename, extensions):
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir, file_type=None, file_ext=None):
    paths = []
    assert os.path.exists(dir) and os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    if file_ext is None:
        file_ext = get_file_extensions(file_type)
    else:
        file_ext = [file_ext]
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            for ext in file_ext:
                if fname.endswith(ext):
                    path = os.path.join(root, fname)
                    paths.append(path)

    return paths
