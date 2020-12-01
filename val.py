"""
Validation script for spectra

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
"""
from util.util import save_boxplot
from util.validator import Validator
from options.val_options import ValidationOptions
from models.models import create_model

opt = ValidationOptions().parse()  # get test options

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
# opt.batch_size = 1    # test code only supports batch_size = 1
# opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.

model = create_model(opt)      # create a model given opt.model and other options

validator = Validator(opt)
err_rel, avg_err_rel, pearson_coefficient = validator.get_validation_score(model)
print('average realative error:', avg_err_rel)
print('pearson coefficient:', pearson_coefficient)
save_boxplot(err_rel, avg_err_rel, opt.results_dir + opt.name, ['Cho', 'NAA'])
print('Done. You can find you the generated validaton plot at', opt.results_dir + opt.name)
