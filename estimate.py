# File to run a test with a pre-trained network. 
# TODO May not be working right now, compare with train

import time
import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.options import TestOptions
# from ray import tune
# from models.auxiliary import progressbar

# Training Phase
opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

print('------------ Creating Training Set ------------')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
# print('#training spectra = %d' % dataset_size)
# print('#training batches = %d' % len(dataset))

print('--------------- Creating Model ---------------')

model = create_model(opt)
print('Model created and initialized')

total_steps = 0
print('------------- Beginning Training -------------')
# for epoch in range(opt.starting_epoch, opt.niter + opt.niter_decay + 1):
    # print('>>>>> Epoch: ', epoch)
epoch_start_time = time.time()

# # Training
for i, data in enumerate(dataset):
    iter_start_time = time.time()
    model.set_input(data)
    model.estimate()

# Saving training results from the current epoch
model.save_spectra(batch=-1)

model.zipFold()

