import math
from models.cycleGAN_W_REG import cycleGAN_W_REG
import os
from models.auxiliaries.physics_model import MRSPhysicsModel
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

# tensorboard --logdir ray_results/
# best config:  {'lambda_A': 11, 'lambda_feat': 0.27359367736275786, 'dlr': 0.00026542079999999994, 'glr': 0.00024000000000000003}

def get_score(validator: Validator, dataset, model: cycleGAN_W_REG):
    avg_abs_err, err_rel, avg_err_rel, r2 = validator.get_validation_score(model, dataset, 20)
    score = np.mean(avg_err_rel)
    return score

def training_function(config, checkpoint_dir=None):
    opt = update_options(init_opt, config)

    physicsModel = MRSPhysicsModel(opt)
    train_set = CreateDataLoader(opt, 'train').load_data()
    val_set = CreateDataLoader(opt, 'val').load_data()

    model = create_model(opt, physicsModel)       # create a model given opt.model and other options
    if checkpoint_dir is not None:
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = model.load_checkpoint(path)
        step = checkpoint['step']
        scores = checkpoint['scores']
    else:
        scores=[]
        step=0

    validator = Validator(opt)

    iter_to_next_display = opt.display_freq
    while True:
        # Loads batch_size samples from the dataset
        for i, data in enumerate(train_set):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            # Only update critic every n_critic steps
            optimize_gen = not(i % opt.n_critic)
            model.optimize_parameters(optimize_G=optimize_gen)   # calculate loss functions, get gradients, update network weights

            iter_to_next_display-= opt.batch_size

            if iter_to_next_display<=0:
                scores.append(get_score(validator, val_set, model))
                step+=1
                iter_to_next_display += opt.display_freq

                if step%STEPS_TO_NEXT_CHECKPOINT == 0 or scores[-1] <= min(scores):
                    with tune.checkpoint_dir(step=step) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        d = {
                            'step': step,
                            'score': scores[-1],
                            'scores': scores
                        }
                        model.create_checkpoint(path, d)

                tune.report(score=scores[-1])
            
            
            
            

class CustomStopper(tune.Stopper):
        '''
        Implements a custom Pleatau Stopper that stops all trails once there was no improvement on the score by more than
        self.tolerance for more than self.patience steps.
        '''
        def __init__(self):
            self.should_stop = False
            # self.max_iter = 350
            self.patience = 70
            self.tolerance = 0.001
            self.scores = []

        def __call__(self, trial_id, result):
            step = result["training_iteration"]-1
            if  len(self.scores)<=step:
                self.scores.append(result["score"])
            else:
                self.scores[step] = min(self.scores[step], result["score"])
            return self.should_stop or (len(self.scores)>self.patience and min(self.scores[-self.patience:]) > min(self.scores[:-self.patience])-self.tolerance)
            # if not self.should_stop and result["score"] < 0.05:
            #     self.should_stop = True
            # return self.should_stop or result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

search_space = {
            "lambda_A":  tune.choice(list(range(8,15,1))),
            "lambda_B":  tune.quniform(1,5,0.5),
            # "lambda_feat": tune.quniform(1,5,0.2),
            "dlr": tune.quniform(0.0001, 0.0003, 0.00002),
            "glr": tune.quniform(0.0001, 0.0003, 0.00002)
        }

PBT = PopulationBasedTraining (
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations=search_space
    )

init_opt = TrainOptions().parse()
os.environ["RAY_MEMORY_MONITOR_ERROR_THRESHOLD"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, init_opt.gpu_ids)))
init_opt.gpu_ids = [0]
stopper = CustomStopper()
BEST_CHECKPOINT_PATH = os.path.join('ray_results/', init_opt.name, 'best')
STEPS_TO_NEXT_CHECKPOINT = 10


analysis = tune.run(
    training_function,
    local_dir='ray_results/',
    name=init_opt.name,
    scheduler=PBT,
    metric="score",
    checkpoint_score_attr="min-score",
    # checkpoint_score_attr="score",
    mode="min",
    stop=stopper,
    export_formats=[ExportFormat.MODEL],
    resources_per_trial={"gpu": 0.11},
    keep_checkpoints_num=1,
    num_samples=40,
    config=search_space,
    raise_on_failed_trial=False
)

# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
x = 0
for d in dfs.values():
    x = max(x,max(d.training_iteration))
    ax = d.plot("training_iteration", "score", ax=ax, legend=False)
x = int(math.ceil(x*1.1/10.0))*10
plt.plot(list(range(x)), [0.15]*x, 'r--')
plt.legend([*['_nolegend_']*len(dfs), '15% error mark'])
plt.xlabel("Steps")
plt.ylabel("Mean Relative Error")
plt.savefig(init_opt.name+'.png', format='png', bbox_inches='tight')

# best_trial = analysis.get_best_logdir(metric="score", mode="min", scope="all")
# best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="score", mode="min")
# print('Best Checkpoint:', best_checkpoint)
# copyfile(best_checkpoint, os.path.join('/home/kreitnerl/mrs-gan/ray_results/', init_opt.name, 'best.pth'))