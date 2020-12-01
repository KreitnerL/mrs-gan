import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

path = '/home/kreitnerl/Datasets/spectra_3_pair/dataset_spectra.mat'
var_name = 'spectra'
save_path = 'spectrum2.png'

spectra = io.loadmat(path)
# spectrum =  spectra[var_name][0][0]
spectrum =  np.sqrt(spectra[var_name][0][0]**2 + spectra[var_name][0][1]**2 )

plt.figure()
plt.plot(spectrum)
plt.savefig(save_path, format='png')

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fake[0])
# plt.savefig('spectrum.png', format='png')