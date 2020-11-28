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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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
            model = MLP(self.save_dir+train, lambda pred, y: self.compute_error(pred, y)[2], gpu=gpu, in_out= (len(train_set.spectra_train[0]), len(self.labels)), batch_size=200)
            if not model.pretrained:
                model.train(train_set.spectra_train, train_set.param_train, test_set.spectra_test, test_set.param_test)
        else:
            raise NotImplementedError()

        predictions = model.predict(test_set.spectra_test)
        err_rel, avg_err_rel, pearson_coefficient = self.compute_error(predictions, test_set.param_test)
        for metabolite in range(len(avg_err_rel)):
            print('Average Relative Error {0}: {1}'.format(self.labels[metabolite], avg_err_rel[metabolite]))
            print('Pearson Coefficient: {0}, {1}'.format(self.labels[metabolite], pearson_coefficient[metabolite]))
        self.save_plot(err_rel, avg_err_rel,  self.save_dir + train + '2' + test + '_' + model_type)

    def compute_error(self, predictions: list, y):
        """
        Compute the realtive errors and the average per metabolite

        Parameters
        ---------
        - predictions: List of predicted quantifications by the random forest
        - y: List of quantifications. N2xM, M = number of metabolites, N2 = number of spectra
        
        Returns
        -------
        - err_rel: List of relative errors. M x N2
        - avg_err_rel: Average relative error per metabolite. M x 1
        """
        err_rel = []
        avg_err_rel = []
        pearson_coefficient = []
        for metabolite in range(len(self.labels)):
            err_rel.append((abs(predictions[:,metabolite] - y[:,metabolite])) / (abs(y[:,metabolite])))
            avg_err_rel.append(np.mean(err_rel[metabolite]))
            pearson_coefficient.append(abs(pearsonr(predictions[:,metabolite], y[:,metabolite])[0]))
        
        return err_rel, avg_err_rel, pearson_coefficient

    def save_plot(self, err_rel, avg_err_rel, path: str):
        """
        Save a boxplot from the given relative errors.

        Parameters
        ---------
        - err_rel: List of relative errors. M x N2
        - path: directory where the plot should be saved.
        """
        max_y = max(np.array(avg_err_rel)+0.15)
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        else:
            plt.figure(self.figure.number)
        plt.boxplot(err_rel, notch = True, labels=self.labels, showmeans=True, meanline=True)
        plt.ylabel('Relative Error')
        plt.title('Error per predicted metabolite')
        plt.gca().set_ylim([0,max_y])
        path = path+'_rel_err_boxplot.png'
        plt.savefig(path, format='png')
        plt.cla()
        print('Saved error plot at', path)

paths = {
    "I": ('/home/kreitnerl/Datasets/spectra_3_pair/dataset_ideal_spectra.mat', '/home/kreitnerl/Datasets/spectra_3_pair/dataset_ideal_quantities.mat', 'spectra'),
    "R": ('/home/kreitnerl/Datasets/spectra_3_pair/dataset_spectra.mat', '/home/kreitnerl/Datasets/spectra_3_pair/dataset_quantities.mat', 'spectra'),
    "UCSF": ('/home/kreitnerl/Datasets/UCSF_TUM_MRSI2/spectra.mat', '/home/kreitnerl/Datasets/UCSF_TUM_MRSI2/quantities.mat', 'spectra'),
    "LCM": ('/home/kreitnerl/Datasets/LCM_MRS/spectra.mat', '/home/kreitnerl/Datasets/LCM_MRS/quantities.mat', 'spectra')
}
gpu = 6

if __name__ == "__main__":
    b = BaselineCreator(save_dir='/home/kreitnerl/mrs-gan/results/baselines/', labels=["cho", "naa"], mag=True, cropping=slice(300, 812), val_split=0.05)
    model = 'MLP'

    # b.create_baseline('I', 'I', model)
    # b.create_baseline('I', 'R', model)
    # b.create_baseline('R', 'R', model)

    # b.create_baseline('I', 'UCSF', model)
    # b.create_baseline('UCSF', 'UCSF', model)

    # b.create_baseline('I', 'LCM', model)
    b.create_baseline('LCM', 'LCM', model)
