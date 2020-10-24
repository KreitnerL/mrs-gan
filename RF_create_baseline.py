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
real_path = '/home/kreitnerl/Datasets/Synthetic_data/dataset_magnitude.mat'
real_parameter_path = '/home/kreitnerl/Datasets/Synthetic_data/dataset_parameters.mat'
real_var_name = 'mag'
save_dir = './results/'

labels = ["cho", "naa"]
val_split = 0.1

def load_dataset(path, var_name, param_path):
    print('load spectra from:', path)
    data = np.array(io.loadmat(path)[var_name]).squeeze()
    
    print('load parameters from:', param_path)
    params = []
    for label in labels:
        params.append(np.array(io.loadmat(param_path)[label]).squeeze())
    return data, np.transpose(params)

def baseline_1():
    real, real_params = load_dataset(real_path, real_var_name, real_parameter_path)
    # Split real data into training / test dataset

    val_split = 0.1
    num_train = val_split * len(real)
    real_train = np.array([real[i] for i in range(num_train)])
    real_test = np.array([real[i] for i in range(num_train, len(real))])
    real_param_train = np.array([real_params[i] for i in range(num_train)])
    real_param_test = np.array([real_params[i] for i in range(num_train, len(real))])
    train_val(real_train, real_test, real_param_train, real_param_test, labels, save_dir)

def baseline_2():
    synthetic, synthetic_params = load_dataset(synthetic_path, synthetic_var_name, synthetic_parameter_path)
    real, real_params = load_dataset(real_path, real_var_name, real_parameter_path)

    # Only use subset of synthetic data
    num_train = 1000
    real_test = np.array([real[i] for i in range(num_train, len(real))])
    real_param_test = np.array([real_params[i] for i in range(num_train, len(real))])
    synthetic_train = np.array([synthetic[i] for i in range(num_train)])
    synthetic_param_train = np.array([synthetic_params[i] for i in range(num_train)])
    train_val(synthetic_train, real_test, synthetic_param_train, real_param_test, labels, save_dir)

def baseline_3():
    synthetic, synthetic_params = load_dataset(synthetic_path, synthetic_var_name, synthetic_parameter_path)

    # Split synthetic data into training / test dataset
    val_split = 0.1
    # num_train = val_split * len(synthetic)
    num_train = 1000
    num_test = 1000
    synthetic_train = np.array([synthetic[i] for i in range(num_train)])
    synthetic_test = np.array([synthetic[i] for i in range(num_train, num_train+num_test)])
    synthetic_param_train = np.array([synthetic_params[i] for i in range(num_train)])
    synthetic_param_test = np.array([synthetic_params[i] for i in range(num_train, num_train+num_test)])
    train_val(synthetic_train, synthetic_test, synthetic_param_train, synthetic_param_test, labels, save_dir)

# baseline_1()
# baseline_2()
baseline_3()

