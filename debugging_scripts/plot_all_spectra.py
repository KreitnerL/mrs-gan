import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# path = '/home/kreitnerl/Datasets/syn_4_real/dataset_spectra.mat'
# path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/spectra.mat'
path = "/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data.mat"
var_name = 'spectra'
ppm_range = [7.171825,-0.501875]
crop_range=slice(300,812)
# crop_range=slice(457,713)
# crop_range=slice(210,722)
x = np.linspace(*ppm_range, 1024)[crop_range]

spectra = io.loadmat(path)[var_name]
spectra[:,:,713:]=0
spectra = spectra[:,:,crop_range]
# spectra = spectra/np.amax(abs(spectra))
plt.figure()
# spectrum =  spectra[var_name][4][0]
mag = np.sqrt(spectra[:,0]**2 + spectra[:,1]**2)
mag = mag/np.amax(mag)

plt.plot(x, mag.transpose())
plt.xlim([x[0], x[-1]])
plt.xlabel('ppm')
plt.title('UCSF Dataset Magnitude Spectra')
# plt.ylim(-1,1)
plt.savefig('UCSF_ALL.png', format='png', bbox_inches='tight')