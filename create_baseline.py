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


def load_dataset(path, param_path, var_name, labels, mag, cropping):
    print('load spectra from:', path)
    data = np.array(io.loadmat(path)[var_name])
    if data.ndim == 2:
        data = np.expand_dims(data, 1)
    
    if mag:
        data = np.sqrt(data[:,0:1,cropping]**2 + data[:,1:2,cropping]**2)
    data = normalize(data).squeeze()

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
    def __init__(self, save_dir, labels, mag=True, cropping=(slice(None,None)), val_split=0.1):
        self.save_dir = save_dir
        self.labels = labels
        self.mag = mag
        self.cropping = cropping
        self.val_split = val_split
        self.datasets: dict[str, Dataset] = dict()

    def get_dataset(self, label: str):
        if label not in self.datasets:
            dataset = Dataset()
            spectra, params = load_dataset(*paths[label], self.labels, self.mag, self.cropping)
            num_test = round(self.val_split * len(spectra))
            num_train = len(spectra) - num_test
            dataset.spectra_train = np.array([spectra[i] for i in range(num_train)])
            dataset.param_train = np.array([params[i] for i in range(num_train)])
            dataset.spectra_test = np.array([spectra[i] for i in range(num_train, num_train+num_test)])
            dataset.param_test = np.array([params[i] for i in range(num_train, num_train+num_test)])
            self.datasets[label] = dataset

        return self.datasets[label]

    def create_baseline(self, train: str, test: str, model_type:str):
        print('Creating baseline:', train, 'to', test)
        train_set: Dataset = self.get_dataset(train)
        test_set: Dataset = self.get_dataset(test)

        if model_type=='RF':
            model = RandomForest(100, self.labels, self.save_dir+train)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train)
        elif model_type=='MLP':
            model = MLP(self.save_dir+train, lambda pred, y: compute_error(pred, y)[2], gpu=gpu, in_out= (len(train_set.spectra_train[0]), len(self.labels)), batch_size=200)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train, test_set.spectra_test, test_set.param_test)
        else:
            raise NotImplementedError()

        predictions = model.predict(test_set.spectra_test)
        err_rel, avg_err_rel, pearson_coefficient = compute_error(predictions, test_set.param_test)
        for metabolite in range(len(avg_err_rel)):
            print('Average Relative Error {0}: {1}'.format(self.labels[metabolite], avg_err_rel[metabolite]))
            print('Pearson Coefficient: {0}, {1}'.format(self.labels[metabolite], pearson_coefficient[metabolite]))
        save_boxplot(err_rel, avg_err_rel,  self.save_dir + train + '2' + test + '_' + model_type, self.labels)


paths = {
    "I": ('/home/kreitnerl/Datasets/spectra_4_pair/dataset_ideal_spectra.mat', '/home/kreitnerl/Datasets/spectra_4_pair/dataset_ideal_quantities.mat', 'spectra'),
    "R": ('/home/kreitnerl/Datasets/spectra_4_pair/dataset_spectra.mat', '/home/kreitnerl/Datasets/spectra_4_pair/dataset_quantities.mat', 'spectra'),
    "UCSF": ('/home/kreitnerl/Datasets/UCSF_TUM_MRSI2/spectra.mat', '/home/kreitnerl/Datasets/UCSF_TUM_MRSI2/quantities.mat', 'spectra'),
    "LCM": ('/home/kreitnerl/Datasets/LCM_MRS/spectra.mat', '/home/kreitnerl/Datasets/LCM_MRS/quantities.mat', 'spectra')
}
gpu = 6

if __name__ == "__main__":
    b = BaselineCreator(save_dir='/home/kreitnerl/mrs-gan/results/baselines/', labels=["cho", "naa"], mag=True, cropping=slice(300, 812), val_split=0.05)
    model = 'MLP'

    # b.create_baseline('I', 'I', model)
    b.create_baseline('I', 'R', model)
    # b.create_baseline('R', 'R', model)

    # b.create_baseline('I', 'UCSF', model)
    # b.create_baseline('UCSF', 'UCSF', model)

    # b.create_baseline('I', 'LCM', model)
    # b.create_baseline('LCM', 'LCM', model)
