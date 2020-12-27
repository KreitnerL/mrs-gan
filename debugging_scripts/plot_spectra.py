import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# path = '/home/kreitnerl/Datasets/syn_4_real/dataset_spectra.mat'
path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/spectra.mat'
# path = '/home/kreitnerl/Datasets/syn_ucsf_corrected2/dataset_spectra.mat'
var_name = 'spectra'
ppm_range = [7.171825,-0.501875]
crop_range_UCSF=slice(200,712)
crop_range = slice(300,812)
x = np.linspace(*ppm_range, 1024)[slice(300,812)]

spectra = io.loadmat(path)[var_name]
spectra[:,:,630:]=0
spectra = spectra[:,:,crop_range_UCSF]
plt.figure()
for i in range(10):
    spectrum = spectra[i]/np.amax(abs(spectra[i]))
    for j in (0,1):
        plt.plot(x, spectrum[j])
    plt.xlim([x[0], x[-1]])
    plt.xlabel('ppm')
    plt.ylim(-1,1)
    plt.savefig('ucsf_ideal/UCSFspectrum%d.png'%i, format='png')
    plt.cla()

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fake[0])
# plt.savefig('spectrum.png', format='png')