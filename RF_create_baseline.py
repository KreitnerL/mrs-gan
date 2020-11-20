"""
Run this file to create a baseline for the performance of the random forest.
The program will test:
    - Train on real, test of real
    - Train on ideal, test on real
    - Train on ideal, test on ideal
"""
import scipy.io as io
from random_forest.random_forest import train_val
import numpy as np

ideal_path = '/home/kreitnerl/Datasets/spectra_3_pair/dataset_ideal_magnitude.mat'
ideal_parameter_path = '/home/kreitnerl/Datasets/spectra_3_pair/dataset_ideal_quantities.mat'
ideal_var_name = 'mag'
real_path = '/home/kreitnerl/Datasets/spectra_3_pair/dataset_magnitude.mat'
real_parameter_path = '/home/kreitnerl/Datasets/spectra_3_pair/dataset_quantities.mat'
real_var_name = 'mag'
fakes_path = '/home/kreitnerl/mrs-gan/results/spec_cyc_entropy_2/fakes.mat'
fakes_parameter_path = '/home/kreitnerl/Datasets/paired_samples/dataset_quantities.mat'
fakes_var_name = 'spectra'
fakes_parameter_offset = 0.1
crop_start = 300
crop_end = 800


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
    if data.ndim == 2:
        data = np.expand_dims(data, 1)
    data = normalize(data).squeeze()
    data = data[:,crop_start:crop_end]
    print('load parameters from:', param_path)
    params = []
    for label in labels:
        params.append(np.array(io.loadmat(param_path)[label]).squeeze())
    return data, np.transpose(params)

class BaselineCreator:
    def __init__(self):
        ideal, ideal_params = load_dataset(ideal_path, ideal_var_name, ideal_parameter_path)
        num_test_ideal = round(val_split * len(ideal))
        num_train_ideal = len(ideal) - num_test_ideal
        self.ideal_train = np.array([ideal[i] for i in range(num_train_ideal)])
        self.ideal_param_train = np.array([ideal_params[i] for i in range(num_train_ideal)])
        self.ideal_test = np.array([ideal[i] for i in range(num_train_ideal, num_train_ideal+num_test_ideal)])
        self.ideal_param_test = np.array([ideal_params[i] for i in range(num_train_ideal, num_train_ideal+num_test_ideal)])

        real, real_params = load_dataset(real_path, real_var_name, real_parameter_path)
        num_test_real = round(val_split * len(real))
        num_train_real = len(real) - num_test_real
        self.real_train = np.array([real[i] for i in range(num_train_real)])
        self.real_param_train = np.array([real_params[i] for i in range(num_train_real)])
        self.real_test = np.array([real[i] for i in range(num_train_real, num_train_real+num_test_real)])
        self.real_param_test = np.array([real_params[i] for i in range(num_train_real, num_train_real+num_test_real)])

    def create_baselines(self):
        print('Creating baseline 1: real to real')
        train_val(self.real_train, self.real_test, self.real_param_train, self.real_param_test, labels, save_dir+'R2R_crop', save_dir+'R_crop_2.joblib')

        print('Creating baseline 2: ideal to real')
        train_val(self.ideal_train, self.real_test, self.ideal_param_train, self.real_param_test, labels, save_dir+'I2R_crop', save_dir+'I_crop_2.joblib')

        print('Creating baseline 3: ideal to ideal')
        train_val(self.ideal_train, self.ideal_test, self.ideal_param_train, self.ideal_param_test, labels, save_dir+'I2I_crop', save_dir+'I_crop_2.joblib')

class Tester():
    def __init__(self):
        self.fakes, self.fake_params = load_dataset(fakes_path, fakes_var_name, fakes_parameter_path)
        offset = round(fakes_parameter_offset*len(self.fake_params))
        self.fake_params = np.array([self.fake_params[i] for i in range(offset, offset+len(self.fakes))])

    def test(self):
        print('Testing pretrained RF on fakes...')
        train_val(None, self.fakes, None, self.fake_params, labels, save_dir+'fake', save_dir+'I.joblib')

if __name__ == "__main__":
    b = BaselineCreator()
    b.create_baselines()
    # t = Tester()
    # t.test()

