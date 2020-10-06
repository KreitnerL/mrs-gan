import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()

print('------------ Creating Training Set ------------')
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training spectra = %d' % dataset_size)
print('#training batches = %d' % len(dataset))

print('--------------- Creating Model ---------------')
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

print('------------- Beginning Training -------------')
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    print('>>>>> Epoch: ', epoch)
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize if i < (len(dataset) - 1) else (dataset_size * (epoch)) - total_steps
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        # if total_steps % opt.save_latest_freq == 0:
        #     print('saving the latest model (epoch %d, total_steps %d)' %
        #           (epoch, total_steps))
        #     model.save('latest')

    model.update_learning_rate()    # update learning rates in the end of every epoch.

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    # Saving training results from the current epoch
    # TODO
    # model.save_spectra(epoch, batch=5)
    # print('saving the latest images (epoch %d, iters %d)' % (epoch, total_steps))

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
