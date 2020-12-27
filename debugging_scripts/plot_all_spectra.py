import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# path = '/home/kreitnerl/Datasets/syn_4_real/dataset_spectra.mat'
path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/spectra.mat'
# path = '/home/kreitnerl/Datasets/syn_ucsf_corrected2/dataset_spectra.mat'
var_name = 'spectra'
ppm_range = [7.171825,-0.501875]
crop_range=slice(210,722)
x = np.linspace(*ppm_range, 1024)[slice(300,812)]

spectra = io.loadmat(path)[var_name]
spectra[:,:,624:]=0
spectra = spectra[:,:,crop_range]
spectra = spectra/np.amax(abs(spectra))
plt.figure()
# spectrum =  spectra[var_name][4][0]
for i in range(len(spectra)):
    spectrum =  np.sqrt(spectra[i][0]**2 + spectra[i][1]**2)
    # spectrum = spectrum/np.amax(abs(spectrum))
    plt.plot(x, spectrum)
plt.xlim([x[0], x[-1]])
plt.xlabel('ppm')
# plt.ylim(-1,1)
plt.savefig('UCSF_all2.png', format='png')