from argparse import Namespace
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

init_opt = TrainOptions().parse()
from ray import tune

def objective(losses: dict):
    return sum([v for v in losses.values()]).detach().cpu().numpy()

def extract_config(config):
    opt = vars(init_opt)
    print('[RayTune]: optimizing', config)
    opt.update(config)
    opt = Namespace(**opt)
    return opt

def training_function(config):
    opt = extract_config(config)

    print('------------ Creating Training Set ------------')
    data_loader = CreateDataLoader(opt)     # get training options
    dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
    dataset_size = len(data_loader)         # get the number of samples in the dataset., 
    print('#training spectra = %d' % dataset_size)
    print('#training batches = %d' % len(dataset))

    print('--------------- Creating Model ---------------')
    model = create_model(opt)       # create a model given opt.model and other options

    print('------------- Beginning Training -------------')
    for epoch in range(opt.n_epochs + opt.n_epochs_decay + 1):
        print('>>>>> Epoch: ', epoch)
        # Loads batch_size samples from the dataset
        for i, data in enumerate(dataset):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # Only update critic every n_critic steps
            optimize_gen = not(i % opt.n_critic)
            model.optimize_parameters(optimize_G=optimize_gen)   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            tune.report(score=objective(losses))

        model.update_learning_rate()    # update learning rates in the end of every epoch.

analysis = tune.run(
    training_function,
    config={
        "dlr": tune.quniform(0.0001, 0.001, 0.0001),
    },
    resources_per_trial={"gpu": 0.5},
    num_samples=6
)
print("best config: ", analysis.get_best_config(metric="score", mode="min", scope='last-10-avg'))
print(analysis.results)