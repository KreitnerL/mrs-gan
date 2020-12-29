"""
Run this file to create a baseline for the performance of the random forest.
The program will test:
    - Train on real, test of real
    - Train on ideal, test on real
    - Train on ideal, test on ideal
"""
import scipy.io as io
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

class Dataset:
    def __init__(self) -> None:
        self.spectra_train = None
        self.param_train = None
        self.spectra_test = None
        self.param_test = None

class BaselineCreator:
    def __init__(self, save_dir, labels, mag=True, val_split=0.1):
        self.save_dir = save_dir
        self.labels = labels
        self.mag = mag
        self.val_split = val_split
        self.datasets: dict[str, Dataset] = dict()

    def get_dataset(self, label: str):
        if label not in self.datasets:
            dataset = Dataset()
            spectra, params = load_dataset(*paths[label], self.labels, self.mag)
            num_test = round(self.val_split * len(spectra))
            num_train = len(spectra) - num_test
            dataset.spectra_train = np.array([spectra[i] for i in range(num_train)])
            dataset.param_train = np.array([params[i] for i in range(num_train)])
            dataset.spectra_test = np.array([spectra[i] for i in range(num_train, num_train+num_test)])
            dataset.param_test = np.array([params[i] for i in range(num_train, num_train+num_test)])
            self.datasets[label] = dataset

        return self.datasets[label]

    def create_baseline(self, train: str, test: str, model_type:str):
        print('------------- Creating baseline:', train, 'to', test, '-------------')
        train_set: Dataset = self.get_dataset(train)
        test_set: Dataset = self.get_dataset(test)

        if model_type=='RF':
            model = RandomForest(100, self.labels, self.save_dir+train)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train)
        elif model_type=='MLP':
            model = MLP(self.save_dir+train, lambda pred, y: np.mean(compute_error(pred, y)[2]), gpu=gpu, in_out= (np.prod(train_set.spectra_train.shape[1:]), len(self.labels)), batch_size=200)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train, test_set.spectra_test, test_set.param_test)
        else:
            raise NotImplementedError()

        predictions = model.predict(test_set.spectra_test)
        mean_abs_err, err_rel, avg_err_rel, r2 = compute_error(predictions, test_set.param_test)
        print('Average Relative Error:', avg_err_rel)
        print('Coefficient of Determination:', r2)
        print('Mean Absolute Error:', mean_abs_err)
        save_boxplot(err_rel, self.save_dir + train + '2' + test + '_' + model_type, self.labels, max_y=max(1.0, np.sum(avg_err_rel)))


paths = {
    "I": ('/home/kreitnerl/Datasets/syn_4_ideal/dataset_spectra.mat', '/home/kreitnerl/Datasets/syn_4_ideal/dataset_quantities.mat', 'spectra', slice(300,812)),
    "R": ('/home/kreitnerl/Datasets/syn_ucsf/dataset_spectra.mat', '/home/kreitnerl/Datasets/syn_ucsf/dataset_quantities.mat', 'spectra', slice(300,812)),
    "UCSF": ('/home/kreitnerl/Datasets/UCSF_TUM_MRSI/spectra_corrected.mat', '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/quantities.mat', 'spectra', slice(210,722)),
    "LCM": ('/home/kreitnerl/Datasets/LCM_MRS/spectra.mat', '/home/kreitnerl/Datasets/LCM_MRS/quantities.mat', 'spectra', slice(210,722))
}
gpu = 6

if __name__ == "__main__":
    b = BaselineCreator(save_dir='/home/kreitnerl/mrs-gan/results/baselines/', labels=["cho", "naa"], mag=False, val_split=0.1)
    model = 'MLP'

    # b.create_baseline('I', 'I', model)
    # b.create_baseline('I', 'R', model)
    b.create_baseline('R', 'R', model)

    b.create_baseline('R', 'UCSF', model)
    b.create_baseline('UCSF', 'UCSF', model)

    # b.create_baseline('I', 'LCM', model)
    # b.create_baseline('LCM', 'LCM', model)
