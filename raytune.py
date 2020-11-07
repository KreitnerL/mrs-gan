from argparse import Namespace
from util.validator import Validator
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from ray import tune
import hyperopt as hp
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1
# export CUDA_VISIBLE_DEVICES=1

# def objective(losses: dict):
#     return sum([v.detach().cpu().numpy() for v in losses.values()])
def report(validator: Validator, model):
    _, avg_err_rel = validator.get_validation_score(model)
    tune.report(err_rate=sum(avg_err_rel))

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

    validator = Validator(opt)

    print('------------- Beginning Training -------------')
    iter_to_next_display = opt.display_freq
    for epoch in range(opt.n_epochs + opt.n_epochs_decay + 1):
        print('>>>>> Epoch: ', epoch)
        # Loads batch_size samples from the dataset
        for i, data in enumerate(dataset):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # Only update critic every n_critic steps
            optimize_gen = not(i % opt.n_critic)
            model.optimize_parameters(optimize_G=optimize_gen)   # calculate loss functions, get gradients, update network weights

            iter_to_next_display-= opt.batch_size
            if iter_to_next_display<=0:
                report(validator, model)
                iter_to_next_display += opt.display_freq
        report(validator, model)

        model.update_learning_rate()    # update learning rates in the end of every epoch.

    report(validator, model)

# Create HyperBand scheduler and minimize err_rate
hyperband = HyperBandScheduler(metric="err_rate", mode="min")
# Specify the search space and maximize err_rate
hyperopt = HyperOptSearch(metric="err_rate", mode="max")

init_opt = TrainOptions().parse()
analysis = tune.run(
    training_function,
    config={
        "dlr": tune.quniform(0.0001, 0.001, 0.0001),
    },
    resources_per_trial={"gpu": 0.3},
    num_samples=6,
    scheduler=hyperband,
    search_alg=hyperopt
)
print("best config: ", analysis.get_best_config(metric="err_rate", mode="min", scope='last-5-avg'))
print(analysis.results)