from util.util import update_options
from util.validator import Validator
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np

# export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1
# export CUDA_VISIBLE_DEVICES=1
# tensorboard --logdir /home/kreitnerl/ray_results

# def objective(losses: dict):
#     return sum([v.detach().cpu().numpy() for v in losses.values()])
def report(validator: Validator, model):
    _, avg_err_rel, pearson_coefficient = validator.get_validation_score(model)
    tune.report(score=np.mean(pearson_coefficient))

def training_function(config):
    opt = update_options(init_opt, config)

    print('------------ Creating Training Set ------------')
    data_loader = CreateDataLoader(opt)     # get training options
    dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options
    dataset_size = len(data_loader)         # get the number of samples in the dataset., 
    opt = update_options(init_opt, {'data_length': dataset.dataset.get_length()})
    print('training spectra = %d' % dataset_size)
    print('training batches = %d' % len(dataset))

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

# Create HyperBand scheduler and maximize score
hyperband = HyperBandScheduler(metric="score", mode="max", max_t=200)
# Specify the search space and maximize score
hyperopt = HyperOptSearch(metric="score", mode="max")

init_opt = TrainOptions().parse()
analysis = tune.run(
    training_function,
    config={
        # "dlr": tune.quniform(0.0002, 0.0005, 0.0001),
        # "glr": tune.quniform(0.0002, 0.0005, 0.0001),
        "lambda_A":  tune.choice(list(range(5,26,5))),
        "lambda_B":  tune.choice(list(range(5,26,5))),
        # "batch_size": tune.choice(list(range(1,100))) 50

        # "which_model_netG": tune.choice([3,4,5,6]), 6
        # "lambda_feat": tune.quniform(0, 5, 0.2) 3
        # "n_downsampling": tune.choice(list(range(2,5))),
        # "n_layers_D": tune.choice(list(range(3,6)))
    },
    resources_per_trial={"gpu": 0.2},
    num_samples=30,
    scheduler=hyperband,
    search_alg=hyperopt,
    raise_on_failed_trial=False
)
print("best config: ", analysis.get_best_config(metric="score", mode="max", scope="last-5-avg"))
print(analysis.results)