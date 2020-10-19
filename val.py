"""
Validation script for spectra

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
import scipy.io as io
from util.util import progressbar
from options.val_options import ValidationOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import numpy as np

opt = ValidationOptions().parse()  # get test options

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
# opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.


print('------------ Creating Test Set ------------')
data_loader = CreateDataLoader(opt)     # get training options
dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
dataset_size = len(data_loader)         # get the number of samples in the dataset.
print('#test spectra = %d' % dataset_size)
print('#test batches = %d' % len(dataset))

print('--------------- Creating Model ---------------')
model = create_model(opt)      # create a model given opt.model and other options

fakes = []
for data in progressbar(dataset, num_iters = opt.num_test):
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    fakes.append(model.get_fake().detach().squeeze(dim=0).cpu().numpy())
fakes = np.array(fakes)
path = str(os.path.join(opt.results_dir, opt.name, 'fakes.mat'))
io.savemat(path, mdict={'spectra': fakes})
print('Done. You can find you the generated spectra at', path)
