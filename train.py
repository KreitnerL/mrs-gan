import time
import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.options import TrainOptions
from util.visualizer import Visualizer
# from ray import tune
from models.auxiliary import progressbar

# Training Phase
opt = TrainOptions().parse()
# visualizer = Visualizer(opt)

print('------------ Creating Training Set ------------')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training spectra = %d' % dataset_size)
print('#training batches = %d' % len(dataset))

print('----------- Creating Validation Set -----------')

opt.phase = 'val'
vdata_loader = CreateDataLoader(opt)
valdataset = vdata_loader.load_data()
vdataset_size = len(vdata_loader)
print('#validation spectra = %d' % vdataset_size)
print('#validation batches = %d' % len(valdataset))


print('--------------- Creating Model ---------------')

total_steps = 0
best = float('inf')
grads = opt.plot_grads
model = create_model(opt)
model.print_buffers()
model.warmup(len(dataset))
print('Model created and initialized')


print('------------- Beginning Training -------------')
for epoch in range(opt.starting_epoch, opt.niter + opt.niter_decay + 1):
    print('>>>>> Epoch: ', epoch)
    epoch_start_time = time.time()
    model.no_plot_grads()

    # # Training
    # model.train()
    for i, data in enumerate(dataset):
        # if i <= int(len(dataset) / 20):
        iter_start_time = time.time()
        prev = total_steps
        total_steps += opt.batchSize if i < (len(dataset) - 1) else (dataset_size * (epoch)) - total_steps
        # epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize(True,epoch)
        model.update_learning_rate(epoch, i)

        # Collecting errors
        if (total_steps % opt.print_freq == 0) or (total_steps % opt.print_freq < prev % opt.print_freq):
            t = (time.time() - iter_start_time) / opt.batchSize
            print('collecting the latest error (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.collect_train_errors()
            # track.log(**results_dict)

        # Plotting errors
        if (total_steps % opt.save_latest_freq == 0) or (total_steps % opt.save_latest_freq < prev % opt.print_freq):
            print('plotting the latest error (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.plot_loss()
            # visualizer.display_current_results(model.collect_images(), epoch)

    # Saving training results from the current epoch
    model.save_spectra(epoch, batch=5)
    print('saving the latest images (epoch %d, total_steps %d)' % (epoch, total_steps))


    # # Validation
    print('>>>>> Validation {}'.format(epoch))
    if (epoch % 5 == 0 or epoch == 1) and grads:
        model.plot_grads(grads)
        print('tracking gradient flow...')
    val_steps = 0
    # model.eval()

    for i, data in enumerate(valdataset):
        # if i <= int(len(valdataset) / 10):
        val_prev = val_steps
        val_steps += opt.batchSize if i < (len(valdataset) - 1) else vdataset_size - val_steps
        model.set_input(data)
        model.optimize(False,epoch)

        # Collecting errors
        if (val_steps % opt.print_freq == 0) or (val_steps % opt.print_freq < val_prev % opt.print_freq):
            model.collect_val_errors()
            print('collecting validation error (epoch %d, val_steps %d)' % (epoch, val_steps))

    val = model.compile_val()
    model.plot_val()
    # track.log(**val_dict)

    # # Assessment and Saving
    if val<best:
        print('saving the model with the best validation score')
        model.best(epoch)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    # visualizer.display_current_results(model.collect_images(), epoch)
    model.save_summary()
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

model.zipFold()

