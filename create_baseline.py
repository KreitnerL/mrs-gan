"""
Run this file to create a the baseline performance for the regression network.
"""
import scipy.io as io
import os
import numpy as np
from validation_networks.random_forest.random_forest import RandomForest
from validation_networks.MLP.MLP import MLP
from util.util import compute_error, save_boxplot
from util.util import normalize


def load_dataset(path, param_path, var_name, roi, labels, mag):
    print('load spectra from:', path)
    data = np.array(io.loadmat(path)[var_name])
    if data.ndim == 2:
        data = np.expand_dims(data, 1)
    
    if mag:
        data = np.sqrt(data[:,0:1,roi]**2 + data[:,1:2,roi]**2)
    else:
        data = data[:,:,roi]
    data = normalize(data)

    print('load parameters from:', param_path)
    params = []
    for label in labels:
        params.append(np.array(io.loadmat(param_path)[label]).squeeze())

    return data, np.transpose(params)

def get_split_indices(dataset_size, val_split=0.2, test_split=0.1, shuffle_data=False):
        """
        Divides the dataset into training, validation and test set.\n
        Returns the indices of samples for train, val and test set.
        """
        if shuffle_data:
            indices = np.random.permutation(dataset_size)
        else:
            indices = np.array(range(dataset_size))
        split1 = round(dataset_size * (1 - val_split - test_split))
        split2 = round(dataset_size * (1 - test_split))

        if test_split==0:
            train_sampler, valid_sampler = indices[:split1], indices[split1:split2]
            test_sampler = np.empty(0)
        else:
            train_sampler, valid_sampler, test_sampler = indices[:split1], indices[split1:split2], indices[split2:]
        return np.sort(train_sampler, axis=0), np.sort(valid_sampler, axis=0), np.sort(test_sampler, axis=0)

class Dataset:
    def __init__(self) -> None:
        self.spectra_train = None
        self.param_train = None
        self.spectra_val = None
        self.param_val = None
        self.spectra_test = None
        self.param_test = None

class BaselineCreator:
    def __init__(self, save_dir, labels, mag=True):
        self.save_dir = save_dir
        self.labels = labels
        self.mag = mag
        self.datasets: "dict[str, Dataset]" = dict()

    def get_dataset(self, label: str):
        if label not in self.datasets:
            dataset = Dataset()
            path, param_path, var_name, roi, val_split, test_split = paths[label]
            spectra, params = load_dataset(path, param_path, var_name, roi, self.labels, self.mag)

            train_ind, val_ind, test_ind = get_split_indices(len(spectra), val_split, test_split)
            dataset.spectra_train, dataset.param_train = spectra[train_ind], params[train_ind]
            dataset.spectra_val, dataset.param_val = spectra[val_ind], params[val_ind]
            dataset.spectra_test, dataset.param_test = spectra[test_ind], params[test_ind]

            self.datasets[label] = dataset
        return self.datasets[label]

    def create_baseline(self, train: str, test: str, model_type:str):
        f.write('\n------------- Creating baseline: ' + train + ' to ' + test + ' -------------\n')
        train_set: Dataset = self.get_dataset(train)
        test_set: Dataset = self.get_dataset(test)

        if model_type=='RF':
            model = RandomForest(100, self.labels, self.save_dir+train)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train)
        elif model_type=='MLP':
            model = MLP(self.save_dir+train, lambda pred, y: np.mean(compute_error(pred, y)[2]), gpu=gpu, in_out= (np.prod(train_set.spectra_train.shape[1:]), len(self.labels)), batch_size=100)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train, test_set.spectra_val, test_set.param_val)
        else:
            raise NotImplementedError()

        f.write('\n----------- Train Set -----------\n')
        predictions = model.predict(test_set.spectra_train)
        mean_abs_err, err_rel, avg_err_rel, r2 = compute_error(predictions, test_set.param_train)
        f.write('Average Relative Error:' + str(list(map(lambda x: round(x, 2), avg_err_rel))) + '\n')
        f.write('Mean Absolute Error:' + str(list(map(lambda x: round(x, 2), mean_abs_err))) + '\n')
        f.write('Coefficient of Determination:' + str(list(map(lambda x: round(x, 2), r2))) + '\n')
        save_boxplot(err_rel, self.save_dir + train + '_to_' + test + '_Train_' + model_type, self.labels, max_y=max(1.0, np.sum(avg_err_rel)))
        
        f.write('\n----------- Val Set -----------\n')
        predictions = model.predict(test_set.spectra_val)
        mean_abs_err, err_rel, avg_err_rel, r2 = compute_error(predictions, test_set.param_val)
        f.write('Average Relative Error:' + str(list(map(lambda x: round(x, 2), avg_err_rel))) + '\n')
        f.write('Mean Absolute Error:' + str(list(map(lambda x: round(x, 2), mean_abs_err))) + '\n')
        f.write('Coefficient of Determination:' + str(list(map(lambda x: round(x, 2), r2))) + '\n')
        save_boxplot(err_rel, self.save_dir + train + '_to_' + test + '_Val_' + model_type, self.labels, max_y=max(1.0, np.sum(avg_err_rel)))

        f.write('\n----------- Test Set -----------\n')
        predictions = model.predict(test_set.spectra_test)
        mean_abs_err, err_rel, avg_err_rel, r2 = compute_error(predictions, test_set.param_test)
        f.write('Average Relative Error:' + str(list(map(lambda x: round(x, 2), avg_err_rel))) + '\n')
        f.write('Mean Absolute Error:' + str(list(map(lambda x: round(x, 2), mean_abs_err))) + '\n')
        f.write('Coefficient of Determination:' + str(list(map(lambda x: round(x, 2), r2))) + '\n')
        save_boxplot(err_rel, self.save_dir + train + '_to_' + test + '_Test_' + model_type, self.labels, max_y=max(1.0, np.sum(avg_err_rel)))


###################################################################################################################################################################################
# Please adapt these options / paths for your environment
###################################################################################################################################################################################
small_crop = slice(457,713)
medium_crop = slice(361,713)
paths = {
    "syn_i_": ('/home/kreitnerl/Datasets/syn_ideal/dataset_spectra.mat', '/home/kreitnerl/Datasets/syn_ideal/dataset_quantities.mat', 'spectra', medium_crop, 0.1, 0.1),
    "syn_r_": ('/home/kreitnerl/Datasets/syn_real/dataset_spectra.mat', '/home/kreitnerl/Datasets/syn_real/dataset_quantities.mat', 'spectra', medium_crop, 0.1, 0.1),
    "syn_ucsf_": ('/home/kreitnerl/Datasets/syn_ucsf/dataset_spectra.mat', '/home/kreitnerl/Datasets/syn_ucsf/dataset_quantities.mat', 'spectra', medium_crop, 0.0601, 0.0668),
    "ucsf_": ('/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data.mat', '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data.mat', 'spectra', medium_crop, 0.0601, 0.0668),
    # "lcm_": ('/home/kreitnerl/Datasets/LCM_MRS/spectra.mat', '/home/kreitnerl/Datasets/LCM_MRS/quantities.mat', 'spectra', slice(210,722), 0.1, 0.1)
}
gpu = 2
dirname = os.path.dirname(__file__)
save_dir = os.path.join(dirname, 'results/baselines/')
f = open(os.path.join(save_dir, 'baselines.txt', 'w'))
###################################################################################################################################################################################

if __name__ == "__main__":
    b = BaselineCreator(save_dir=save_dir, labels=["cho", "naa"], mag=False)
    model = 'MLP'

    b.create_baseline('syn_i_', 'syn_i_', model)
    b.create_baseline('syn_ucsf_', 'syn_ucsf_', model)
    b.create_baseline('syn_r_', 'syn_r_', model)
    b.create_baseline('ucsf_', 'ucsf_', model)
    
    b.create_baseline('syn_r_', 'ucsf_', model)
