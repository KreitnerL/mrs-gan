from functools import reduce
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from util.util import progressbar
plt.figure()

d = io.loadmat('/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_raw_data.mat')
metabolites = ['cre', 'cho', 'naa']
cre = np.squeeze(d['cre'])
cho = np.squeeze(d['cho'])
naa = np.squeeze(d['naa'])
num_spectra_per_patient = np.squeeze(d['num_spectra_per_patient'])
spectra = np.squeeze(d['spectra'])
spectra[:,:,623:]=0
mag = np.sqrt(spectra[:,0]**2 + spectra[:,1]**2)
ppm_range = [7.171825,-0.501875]
crop_range_UCSF=slice(210,722)
x = np.linspace(*ppm_range, 1024)[slice(300,812)]
x2 = np.linspace(0,1023,1024)[crop_range_UCSF]
met_range = {'cre': slice(445,500), 'cho': slice(400,440), 'naa': slice(550,None)}
# met_range = {'cre': slice(250,320), 'cho': slice(400,440), 'naa': slice(550,None)}
peak_heights = {key: np.max(mag[:,val], -1) for key,val in met_range.items()}

threshold = 0.3
patient_offset = 0
absolute_quantities = {key: [] for key in metabolites}
blacklist = []
edge_cases = []
for num_spectra in progressbar(num_spectra_per_patient, 'Patient', 20):
    for metabolite in metabolites:
        best_prediction = (num_spectra, None)
        for j in range(1,10):
            concentrations = list(np.squeeze(d[metabolite])[patient_offset: patient_offset+num_spectra])
            # sorted_peak_heights = sorted(list(peak_heights[metabolite][patient_offset: patient_offset+num_spectra]))
            sorted_met = sorted(concentrations)

            max_index_1 = concentrations.index(sorted_met[-j])
            concentration_1 = concentrations[max_index_1]
            preak_height_1 = peak_heights[metabolite][patient_offset+max_index_1]

            ratio_list = []
            for i in range(40):
                max_index_2 = concentrations.index(sorted_met[-i])
                if max_index_1==max_index_2:
                    continue
                concentration_2 = concentrations[max_index_2]
                peak_height_2 = peak_heights[metabolite][patient_offset+max_index_2]
                ratio_list.append(abs((preak_height_1-peak_height_2)/(concentration_1-concentration_2)))
            ratio_list = np.array(ratio_list)
            ratio = ratio_list.mean()

            valid_map=None
            for i in range(5):
                valid_map = abs((ratio_list-ratio)/ratio)<threshold
                if any(valid_map):
                    ratio = ratio_list[valid_map].mean()
                else:
                    ratio/=2.0

            predicted = preak_height_1-((concentration_1-concentrations)*ratio)
            deviations = (peak_heights[metabolite][patient_offset: patient_offset+num_spectra]-predicted)/predicted
            invalid = sum(abs(deviations) > threshold)
            num_overpred = sum(deviations<0)
            if invalid<best_prediction[0] or (invalid==best_prediction[0] and num_overpred>best_prediction[2]):
                best_prediction = invalid, predicted, num_overpred

        absolute_quantities[metabolite].extend(best_prediction[1])
    patient_offset += num_spectra
relative_cho = np.divide(absolute_quantities['cho'], absolute_quantities['cre'])
relative_naa = np.divide(absolute_quantities['naa'], absolute_quantities['cre'])

deviations = {metabolite: abs((np.max(mag[:,met_range[metabolite]],-1)-absolute_quantities[metabolite][:])/absolute_quantities[metabolite][:]) for metabolite in metabolites}
max_deviations = list(map(max, zip(deviations['cre'], deviations['cho'], deviations['naa'])))
whitelist = np.array(max_deviations) < threshold

conditions = [
    relative_cho>0,
    relative_cho<3.5,
    relative_naa>0,
    relative_naa<3.5,
    whitelist
]
valid_indices = reduce(lambda x, y: x&y, conditions)

# sort out outliers
relative_cho = relative_cho[valid_indices]
relative_naa = relative_naa[valid_indices]
spectra = spectra[valid_indices]
sorted_z = sorted(np.array(max_deviations)[valid_indices])
for i in range(1,10):
    ind = max_deviations.index(sorted_z[-i])
    concentrations = 'cre' if deviations['cre'][ind]==sorted_z[-i] else ('cho' if deviations['cho'][ind]==sorted_z[-i] else 'naa')
    mag_val = peak_heights[concentrations][ind]
    pred = np.array(absolute_quantities[concentrations])[ind]
    plt.cla()
    plt.plot(x2, mag[ind][crop_range_UCSF])
    plt.plot(x2, [mag_val]*512)
    plt.plot(x2, [pred]*512)
    plt.legend(['_nolegend_','%s peak'%concentrations, '%s prediction'%concentrations])
    plt.title(concentrations + ': ' + str(mag_val) + ' vs ' + str(pred))
    plt.savefig('edge_case%d.png'%i)

sorted_z = sorted(np.array(max_deviations)[~valid_indices])
for i in range(1,10):
    ind = max_deviations.index(sorted_z[-i])
    concentrations = 'cre' if deviations['cre'][ind]==sorted_z[-i] else ('cho' if deviations['cho'][ind]==sorted_z[-i] else 'naa')
    mag_val = peak_heights[concentrations][ind]
    pred = np.array(absolute_quantities[concentrations])[ind]
    plt.cla()
    plt.plot(x2, mag[ind][crop_range_UCSF])
    plt.plot(x2, [mag_val]*512)
    plt.plot(x2, [pred]*512)
    plt.legend(['_nolegend_','%s peak'%concentrations, '%s prediction'%concentrations])
    plt.title(concentrations + ': ' + str(mag_val) + ' vs ' + str(pred))
    plt.savefig('worst_cases%d.png'%i)

patient_offset=0
for i, num_patients in enumerate(num_spectra_per_patient):
    num_spectra_per_patient[i] = sum(valid_indices[patient_offset:patient_offset+num_patients])
    patient_offset+=num_patients
assert sum(num_spectra_per_patient) == len(relative_cho)


plt.figure()
plt.hist(relative_cho, bins=100, histtype='barstacked')
plt.title('cho distribution')
plt.savefig('cho_all.png')
plt.figure()
plt.title('NAA distribution')
plt.hist(relative_naa, bins=100, histtype='barstacked')
plt.savefig('naa_all.png')

spectra = np.roll(spectra, 90, axis=-1)

path = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data_2.mat'
io.savemat(path, {'cho': relative_cho, 'naa': relative_naa, 'spectra': spectra, 'num_spectra_per_patient': num_spectra_per_patient}, do_compression=True)
print('Done. You can find the generated mat file at', path)