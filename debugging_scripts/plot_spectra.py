import matplotlib.pyplot as plt
import scipy.io as io

path = '/home/kreitnerl/Datasets/spectra_3/dataset_magnitude.mat'
var_name = 'mag'
save_path = 'spectrum2.png'

spectra = io.loadmat(path)
spectrum = spectra[var_name][0][0]

plt.figure()
plt.plot(spectrum)
plt.savefig(save_path, format='png')