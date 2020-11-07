"""
Run this file to create a baseline for the performance of the random forest.
The program will test:
    - Train on real, Test of real
    - Train on synthetic, test on real
    - Train on synthetic, test on synthetic
"""
import scipy.io as io
from random_forest.random_forest import train_val
import numpy as np


synthetic_path = '/home/kreitnerl/Datasets/Synthetic_data/dataset_magnitude.mat'
synthetic_parameter_path = '/home/kreitnerl/Datasets/Synthetic_data/dataset_parameters.mat'
synthetic_var_name = 'mag'
real_path = '/home/kreitnerl/Datasets/second_distribution_dataset/dataset_magnitude.mat'
real_parameter_path = '/home/kreitnerl/Datasets/second_distribution_dataset/dataset_parameters.mat'
real_var_name = 'mag'
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
        
        real, real_params = load_dataset(real_path, real_var_name, real_parameter_path)
        num_test_real = int(val_split * len(real))
        num_train_real = len(real) - num_test_real
        self.real_train = np.array([real[i] for i in range(num_train_real)])
        self.real_param_train = np.array([real_params[i] for i in range(num_train_real)])
        self.real_test = np.array([real[i] for i in range(num_train_real, num_train_real+num_test_real)])
        self.real_param_test = np.array([real_params[i] for i in range(num_train_real, num_train_real+num_test_real)])

        synthetic, synthetic_params = load_dataset(synthetic_path, synthetic_var_name, synthetic_parameter_path)
        num_test_syn = int(val_split * len(synthetic))
        num_train_syn = len(synthetic) - num_test_syn
        self.synthetic_train = np.array([synthetic[i] for i in range(num_train_syn)])
        self.synthetic_param_train = np.array([synthetic_params[i] for i in range(num_train_syn)])
        self.synthetic_test = np.array([synthetic[i] for i in range(num_train_syn, num_train_syn+num_test_syn)])
        self.synthetic_param_test = np.array([synthetic_params[i] for i in range(num_train_syn, num_train_syn+num_test_syn)])


    def create_baselines(self):
        print('Creating baseline 1: Real to Real')
        train_val(self.real_train, self.real_test, self.real_param_train, self.real_param_test, labels, save_dir+'baseline1/')
        print('Creating baseline 2: Syntethic to Real')
        train_val(self.synthetic_train, self.real_test, self.synthetic_param_train, self.real_param_test, labels, save_dir+'baseline2/')
        print('Creating baseline 3: Syntethic 2 Syntethic')
        train_val(self.synthetic_train, self.synthetic_test, self.synthetic_param_train, self.synthetic_param_test, labels, save_dir+'baseline3/')

if __name__ == "__main__":
    b = BaselineCreator()
    b.create_baselines()

