"""
Validation script for spectra

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
from models.auxiliaries.physics_model import PhysicsModel
from util.util import load_options, merge_options, save_boxplot
from util.validator import Validator
from options.val_options import ValidationOptions
from models.models import create_model
import os

validationOptions = ValidationOptions()
opt = validationOptions.parse()  # get test options
train_options = load_options(os.path.join(opt.checkpoints_dir, opt.name, 'opt.txt'))
default_options = validationOptions.get_defaults()
opt = merge_options(default_options, train_options, opt)

# hard-code some parameters for test
opt.phase = 'val'
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.

validator = Validator(opt)
physicsModel = PhysicsModel(opt)
model = create_model(opt, physicsModel)      # create a model given opt.model and other options

avg_abs_err, err_rel, avg_err_rel, r2 = validator.get_validation_score(model)
print('average realative error:', avg_err_rel)
print('pearson coefficient:', r2)
save_boxplot(err_rel, avg_err_rel, opt.results_dir + opt.name, opt.label_names)
print('Done. You can find you the generated validaton plot at', opt.results_dir + opt.name)
