"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer, save_images
from util import html

opt = TestOptions().parse()  # get test options

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


visualizer = Visualizer(opt)    # create a visualizer that display/save images and plots
# create a website
web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch_count))  # define the website directory
print('creating web directory', web_dir)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch_count))
for i, data in enumerate(dataset):
    if i >= opt.num_test:  # only apply our model to opt.num_test images.
        break
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    image_paths = model.get_image_paths()
    if i % opt.print_freq == 0:  # save images to an HTML file
        print('processing (%04d)-th image...' % (i))
        visualizer.display_current_results(model.get_current_visuals())
    save_images(webpage, visuals, image_paths, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
webpage.save()  # save the HTML
