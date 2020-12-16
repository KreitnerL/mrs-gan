"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
from argparse import Namespace
from data.dicom_spectral_dataset import DicomSpectralDataset
import os
from util.util import progressbar
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer, save_images
from util import html
import numpy as np
import scipy.io as io

opt = TestOptions().parse()  # get test options

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.


print('------------ Creating Test Set ------------')
data_loader = CreateDataLoader(opt)     # get training options
dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
dataset_size = len(data_loader)         # get the number of samples in the dataset.
print('test spectra = %d' % dataset_size)
print('test batches = %d' % len(dataset))
opt.data_length=dataset.dataset.get_length()

model = create_model(opt)      # create a model given opt.model and other options


visualizer = Visualizer(opt)    # create a visualizer that display/save images and plots
# create a website
web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch_count))  # define the website directory
print('creating web directory', web_dir)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch_count))
fakes=[]
for data in progressbar(dataset, num_iters = opt.num_test):
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    fakes.append(model.get_fake().detach().squeeze(dim=0).cpu().numpy())
    visuals = model.get_current_visuals()  # get image results
    image_paths = model.get_image_paths()
    save_images(webpage, visuals, image_paths, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
fakes = np.squeeze(np.array(fakes))
io.savemat(opt.results_dir + opt.name + '/fakes.mat', {"spectra": fakes})
webpage.save()  # save the HTML
