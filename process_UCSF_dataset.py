from functools import reduce
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
plt.figure()

d = io.loadmat('/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data_raw.mat')
metabolites = ['cre', 'cho', 'naa']
cre_abs = np.squeeze(d['cre'])
cho_abs = np.squeeze(d['cho'])
naa_abs = np.squeeze(d['naa'])
num_spectra_per_patient = np.squeeze(d['num_spectra_per_patient'])
spectra = np.squeeze(d['spectra'])

cho = cho_abs/cre_abs
naa = naa_abs/cre_abs

conditions = [
    cho>0,
    cho<3.5,
    naa>0,
    naa<3.5
]
valid_indices = reduce(lambda x, y: x&y, conditions)

cho = cho[valid_indices]
naa = naa[valid_indices]
spectra = spectra[valid_indices]

spectra = np.roll(spectra, 90, axis=-1)
patient_offset=0
for i, num_patients in enumerate(num_spectra_per_patient):
    num_spectra_per_patient[i] = sum(valid_indices[patient_offset:patient_offset+num_patients])
    patient_offset+=num_patients
assert sum(num_spectra_per_patient) == len(cho)

plt.figure()
plt.hist(cho, bins=100, histtype='barstacked')
plt.title('cho distribution')
plt.savefig('cho_all.png')
plt.figure()
plt.title('NAA distribution')
plt.hist(naa, bins=100, histtype='barstacked')
plt.savefig('naa_all.png')


path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data_corrected.mat'
io.savemat(path, {'cho': cho, 'naa': naa, 'spectra': spectra, 'num_spectra_per_patient': num_spectra_per_patient}, do_compression=True)
print('Done. You can find the generated mat file at', path)