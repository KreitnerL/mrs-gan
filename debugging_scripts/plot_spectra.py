import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

# path = '/home/kreitnerl/Datasets/syn_4_real/dataset_spectra.mat'
path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/spectra.mat'
# path = '/home/kreitnerl/Datasets/ucsf_test/dataset_spectra.mat'
var_name = 'spectra'

spectra = io.loadmat(path)
# spectrum =  spectra[var_name][4][0]
for i in range(10):
    spectrum =  np.sqrt(spectra[var_name][i][0]**2 + spectra[var_name][i][1]**2)
    spectrum = spectrum/np.amax(spectrum)
    plt.figure()
    plt.plot(spectrum)
    plt.savefig('ucsf_ideal/UCSFspectrum%d.png'%i, format='png')

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fake[0])
# plt.savefig('spectrum.png', format='png')