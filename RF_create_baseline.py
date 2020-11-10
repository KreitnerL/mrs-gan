"""
Run this file to create a baseline for the performance of the random forest.
The program will test:
    - Train on target, test of target
    - Train on source, test on target
    - Train on source, test on source
"""
import scipy.io as io
from random_forest.random_forest import train_val
import numpy as np


source_path = '/home/kreitnerl/Datasets/updated_dataset/dataset_magnitude.mat'
source_parameter_path = '/home/kreitnerl/Datasets/updated_dataset/dataset_quantities.mat'
source_var_name = 'mag'
target_path = '/home/kreitnerl/Datasets/second_distribution_dataset/dataset_magnitude.mat'
target_parameter_path = '/home/kreitnerl/Datasets/second_distribution_dataset/dataset_quantities.mat'
target_var_name = 'mag'
save_dir = './results/'

labels = ["cho", "naa"]
val_split = 0.1

def normalize(spectra):
    max_per_spectrum = np.amax(abs(spectra),(1,2))
    max_per_spectrum = np.repeat(max_per_spectrum[:, np.newaxis], spectra.shape[1], axis=1)
    max_per_spectrum = np.repeat(max_per_spectrum[:, :, np.newaxis], spectra.shape[2], axis=2)
    return np.divide(spectra, max_per_spectrum)

def load_dataset(path, var_name, param_path):
    print('load spectra from:', path)
    data = np.array(io.loadmat(path)[var_name])
    data = normalize(data).squeeze()
    
    print('load parameters from:', param_path)
    params = []
    for label in labels:
        params.append(np.array(io.loadmat(param_path)[label]).squeeze())
    return data, np.transpose(params)


class BaselineCreator:
    def __init__(self):
        
        target, target_params = load_dataset(target_path, target_var_name, target_parameter_path)
        num_test_target = int(val_split * len(target))
        num_train_target = len(target) - num_test_target
        self.target_train = np.array([target[i] for i in range(num_train_target)])
        self.target_param_train = np.array([target_params[i] for i in range(num_train_target)])
        self.target_test = np.array([target[i] for i in range(num_train_target, num_train_target+num_test_target)])
        self.target_param_test = np.array([target_params[i] for i in range(num_train_target, num_train_target+num_test_target)])

        source, source_params = load_dataset(source_path, source_var_name, source_parameter_path)
        num_test_syn = int(val_split * len(source))
        num_train_syn = len(source) - num_test_syn
        self.source_train = np.array([source[i] for i in range(num_train_syn)])
        self.source_param_train = np.array([source_params[i] for i in range(num_train_syn)])
        self.source_test = np.array([source[i] for i in range(num_train_syn, num_train_syn+num_test_syn)])
        self.source_param_test = np.array([source_params[i] for i in range(num_train_syn, num_train_syn+num_test_syn)])


    def create_baselines(self):
        print('Creating baseline 1: target to target')
        train_val(self.target_train, self.target_test, self.target_param_train, self.target_param_test, labels, save_dir+'T2T', save_dir+'T2T.joblib')
        print('Creating baseline 2: source to target')
        train_val(self.source_train, self.target_test, self.source_param_train, self.target_param_test, labels, save_dir+'S2T', save_dir+'S2T.joblib')
        print('Creating baseline 3: source to source')
        train_val(self.source_train, self.source_test, self.source_param_train, self.source_param_test, labels, save_dir+'S2S', save_dir+'S2T.joblib')

if __name__ == "__main__":
    b = BaselineCreator()
    b.create_baselines()

