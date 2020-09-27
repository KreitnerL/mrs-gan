import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.shuffle = False  # no shuffle
opt.no_flip = True  # no flip

print('------------ Creating Validation Set ------------')
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#validation spectra = %d' % dataset_size)
print('#validation batches = %d' % len(dataset))

print('--------------- Creating Model ---------------')
model = create_model(opt)
print('Model created and initialized')


best = float('inf')
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    # TODO
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
