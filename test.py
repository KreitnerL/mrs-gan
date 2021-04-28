"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
from debugging_scripts.visualize_results import generate_images_of_spectra
from models.auxiliaries.physics_model import MRSPhysicsModel
import os
from util.util import load_options, merge_options, mkdir, progressbar
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
import scipy.io as io

# opt = TestOptions().parse()  # get test options

testOptions = TestOptions()
opt = testOptions.parse()  # get test options
train_options = load_options(os.path.join(opt.checkpoints_dir, opt.name, 'opt.txt'))
default_options = testOptions.get_defaults()
opt = merge_options(default_options, train_options, opt)

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1

pysicsModel = MRSPhysicsModel(opt)
data_loader = CreateDataLoader(opt, 'test')     # get training options
dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
dataset_size = len(data_loader)         # get the number of samples in the dataset.

model = create_model(opt, pysicsModel)      # create a model given opt.model and other options
model.load_checkpoint(opt.model_path)

items_list = None
for data in progressbar(dataset, num_iters = opt.num_test):
    model.set_input(data)  # unpack data from data loader
    model.test()

    items = model.get_items()
    if items_list is None:
        items_list = items
    else:
        items_list = {key: np.concatenate([val, items[key]], axis=0) for key,val in items_list.items()}
path = os.path.join(opt.results_dir, opt.name, 'items.mat')
mkdir(os.path.dirname(path))
io.savemat(path, items_list)

if opt.num_visuals>0:
    generate_images_of_spectra(items_list, opt.num_visuals, os.path.join(opt.results_dir, opt.name), x = np.linspace(*opt.ppm_range, opt.full_data_length)[opt.roi])
