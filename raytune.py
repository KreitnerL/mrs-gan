import os
from models.auxiliaries.physics_model import PhysicsModel
from util.util import update_options
from util.validator import Validator
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.trial import ExportFormat
import numpy as np
import matplotlib.pyplot as plt


os.environ["RAY_MEMORY_MONITOR_ERROR_THRESHOLD"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1
# export CUDA_VISIBLE_DEVICES=1
# tensorboard --logdir /home/kreitnerl/ray_results

def report(validator: Validator, model):
    avg_abs_err, err_rel, avg_err_rel, r2 = validator.get_validation_score(model)
    print(avg_err_rel)
    tune.report(score=np.mean(avg_err_rel))

def training_function(config, checkpoint_dir=None):
    opt = update_options(init_opt, config)

    physicsModel = PhysicsModel(opt)
    data_loader = CreateDataLoader(opt)     # get training options
    dataset = data_loader.load_data()       # create a dataset given opt.dataset_mode and other options

    model = create_model(opt, physicsModel)       # create a model given opt.model and other options
    if checkpoint_dir is not None:
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = model.load_checkpoint(path)
        step = checkpoint['step']
        score = checkpoint['score']
    else:
        score = 0
        step=0

    validator = Validator(opt)

    iter_to_next_display = opt.display_freq
    iter_to_next_save = 25000
    while True:
        # Loads batch_size samples from the dataset
        for i, data in enumerate(dataset):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # Only update critic every n_critic steps
            optimize_gen = not(i % opt.n_critic)
            model.optimize_parameters(optimize_G=optimize_gen)   # calculate loss functions, get gradients, update network weights

            iter_to_next_save -= opt.batch_size
            if iter_to_next_save<=0:
                with tune.checkpoint_dir(step=step) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    d = {
                        'step': step,
                        'score': score
                    }
                    model.create_checkpoint(path, d)
                iter_to_next_save=25000
            
            iter_to_next_display-= opt.batch_size
            if iter_to_next_display<=0:
                report(validator, model)
                step+=1
                iter_to_next_display += opt.display_freq
            

class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 350
            if not self.should_stop and result["score"] < 0.05:
                self.should_stop = True
            return self.should_stop or result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

search_space = {
            "lambda_A":  tune.choice(list(range(8,15,1))),
            # "lambda_B":  tune.quniform(1,5,0.5), 2
            "lambda_feat": tune.quniform(1,5,0.2),
            "dlr": tune.quniform(0.0001, 0.0003, 0.00002),
            "glr": tune.quniform(0.0001, 0.0003, 0.00002)
        }

PBT = PopulationBasedTraining (
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations=search_space
    )

init_opt = TrainOptions().parse()
stopper = CustomStopper()


analysis = tune.run(
    training_function,
    name='pbt_WGP_REG',
    scheduler=PBT,
    metric="score",
    mode="min",
    stop=stopper,
    export_formats=[ExportFormat.MODEL],
    checkpoint_score_attr="score",
    resources_per_trial={"gpu": 0.125},
    keep_checkpoints_num=16,
    num_samples=16,
    config=search_space,
    raise_on_failed_trial=False
)
print("best config: ", analysis.get_best_config(metric="score", mode="min"))

# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
for d in dfs.values():
    ax = d.plot("training_iteration", "score", ax=ax, legend=False)
plt.xlabel("Steps")
plt.ylabel("Relative Absolute Error")
plt.savefig('raytune.png', format='png')








# # Create HyperBand scheduler and minimize score
# hyperband = HyperBandScheduler(metric="score", mode="min", min_t=500)
# # Specify the search space and minimize score
# hyperopt = HyperOptSearch(metric="score", mode="min")
# analysis = tune.run(
#     training_function,
#     name='pbt_WGP_REG',
#     config={
#         # "dlr": tune.quniform(0.0002, 0.0005, 0.0001),
#         # "glr": tune.quniform(0.0002, 0.0005, 0.0001),
#         "lambda_A":  tune.choice(list(range(5,15,5))),
#         "lambda_B":  tune.choice(list(range(5,15,5))),
#         "lambda_feat": tune.choice(list(range(0,10,2)))
#         # "batch_size": tune.choice(list(range(1,100))) 50

#         # "which_model_netG": tune.choice([3,4,5,6]), 6
#         # "lambda_feat": tune.quniform(0, 5, 0.2) 3
#         # "n_downsampling": tune.choice(list(range(2,5))),
#         # "n_layers_D": tune.choice(list(range(3,6)))
#     },
#     resources_per_trial={"gpu": 0.2},
#     num_samples=40,
#     scheduler=hyperband,
#     search_alg=hyperopt,
#     raise_on_failed_trial=False
# )