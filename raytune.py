from models.cycleGAN_W_REG import cycleGAN_W_REG
import os
from models.auxiliaries.mrs_physics_model import MRSPhysicsModel
from util.util import update_options
from util.validator import Validator
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from ray import tune
from ray.tune.schedulers.pb2 import PB2
from ray.tune.trial import ExportFormat
import numpy as np
from util.plot_PBT import plotPBT, plot_pbt_schedule
import torch
import random
SEED = random.randint(0,1e6)
# tensorboard --logdir ray_results/

def get_score(validator: Validator, dataset, model: cycleGAN_W_REG):
    avg_abs_err, err_rel, avg_err_rel, r2, median_rel_err = validator.get_validation_score(model, dataset, 20)
    score = np.mean(median_rel_err)
    return score

def training_function(config, checkpoint_dir=None):
    torch.manual_seed(config['seed'])
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

        def stop_all(self):
            return self.should_stop

def compute_gpu_load(num_trails):
    return {"gpu": int(100.0/num_trails)/100.0 }

search_space = {
            "lambda_A":  [0.,20],
            "lambda_B":  [0.,20.],
            "lambda_feat": [0.,10.],
            "dlr": [0.0001, 0.0004],
            "glr": [0.0001, 0.0004]
        }

PBB = PB2(
    time_attr="training_iteration",
    perturbation_interval=10,
    hyperparam_bounds=search_space
)

init_opt = TrainOptions().parse()
os.environ["RAY_MEMORY_MONITOR_ERROR_THRESHOLD"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, init_opt.gpu_ids)))
init_opt.gpu_ids = [0]
stopper = CustomStopper()
BEST_CHECKPOINT_PATH = os.path.join('ray_results/', init_opt.name, 'best')
STEPS_TO_NEXT_CHECKPOINT = 10

start_config = {key: tune.uniform(*val) for key, val in search_space.items()}
start_config.update({'seed': SEED})

analysis = tune.run(
    training_function,
    local_dir='ray_results/',
    name=init_opt.name,
    scheduler=PBB,
    metric="score",
    checkpoint_score_attr="min-score",
    mode="min",
    stop=stopper,
    export_formats=[ExportFormat.MODEL],
    resources_per_trial=compute_gpu_load(10),
    keep_checkpoints_num=1,
    num_samples=20,
    config=start_config,
    raise_on_failed_trial=False
)

plotPBT(os.path.join('ray_results/', init_opt.name))
plot_pbt_schedule(os.path.join('ray_results/', init_opt.name, ''))