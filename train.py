from argparse import Namespace
from data.dicom_spectral_dataset import DicomSpectralDataset
import time
from util.validator import Validator
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.visdom import Visdom

opt = TrainOptions().parse()

print('------------ Creating Training Set ------------')
data_loader = CreateDataLoader(opt)     # get training options
dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
dataset_size = len(data_loader)         # get the number of samples in the dataset.
print('training spectra = %d' % dataset_size)
print('training batches = %d' % len(dataset))
if isinstance(dataset.dataset, DicomSpectralDataset):
    opt = vars(opt)
    opt.update({'data_length': dataset.dataset.get_length()})
    opt = Namespace(**opt)

model = create_model(opt)       # create a model given opt.model and other options
visualizer = Visualizer(opt)    # create a visualizer that display/save images and plots
visdom = Visdom(opt)

total_iters = 0                 # the total number of training iterations
t_data = 0

if opt.val_path:
    validator = Validator(opt)
else:
    validator = None

print('------------- Beginning Training -------------')
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    print('>>>>> Epoch: ', epoch)
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visdom.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    # Loads batch_size samples from the dataset
    for i, data in enumerate(dataset):
        iter_start_time = time.time()  # timer for computation per iteration

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        # Only update critic every n_critic steps
        optimize_gen = not(i % opt.n_critic)
        model.optimize_parameters(optimize_G=optimize_gen)   # calculate loss functions, get gradients, update network weights
        
        if opt.display_id > 0 and total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            visdom.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            t_data = iter_start_time - iter_data_time
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, total_iters)
            
        if total_iters % opt.plot_freq == 0:
            visualizer.plot_current_losses()

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            if opt.val_path:
                _, avg_err_rel, pearson_coefficient = validator.get_validation_score(model)
                visualizer.plot_current_validation_score(pearson_coefficient, total_iters)
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save(save_suffix)

        iter_data_time = time.time()

    visualizer.save_smooth_loss()
    visdom.display_current_results(model.get_current_visuals(), epoch, True)

    model.update_learning_rate()    # update learning rates in the end of every epoch.

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

if opt.val_path:
    _, avg_err_rel, pearson_coefficient = validator.get_validation_score(model)
    visualizer.plot_current_validation_score(pearson_coefficient, total_iters)
model.save('latest')