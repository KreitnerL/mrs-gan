import pydicom
import numpy as np
import scipy.io as io
import struct

basepath = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/batch_1/tar_bundle/data/p41/jasonc/BjoernRandomTrees/data_to_share/for_tum_t10169/ucsf/t10169_UCSF_'
metabolites = [
    'cre',
    'cho',
    'NAA'
]

metabolic_map_info = pydicom.dcmread(basepath+ metabolites[0] + '.dcm')
dimZ = int(metabolic_map_info.NumberOfFrames)
dimY = int(metabolic_map_info.Columns)
dimX = int(metabolic_map_info.Rows)
num_voxels = dimX*dimY*dimZ

# Stored as a byte array where each value takes 2 bytes!
data = np.array(metabolic_map_info.pixel_array)
metabolic_map = np.flip(np.flip(data, 1), 2)
metabolic_map_flat = metabolic_map.flatten()

# Most of the voxels are not activated. They contain the a fix baseline value
baseline = np.bincount(metabolic_map_flat).argmax()
# Get the index of all activated voxels
activated_index = [i for i, val in enumerate(metabolic_map_flat) if val != baseline]

mat_dict={}
for metabolite in metabolites:
    metabolic_map_info = pydicom.dcmread(basepath+ metabolite + '.dcm')
    data = np.array(metabolic_map_info.pixel_array)
    data = np.array(data).reshape((dimX, dimY, dimZ))
    meta_map = np.flip(np.flip(data, 1), 2).flatten()
    mat_dict[metabolite] = np.array([val for i,val in enumerate(meta_map) if i in activated_index])

spectra_info = pydicom.dcmread(basepath+ 'proc.dcm')
if 'DataPointColumns' in spectra_info:
    dimS = spectra_info.DataPointColumns
else:
    dimS = 2048
spectra_data = struct.unpack('f' * (num_voxels*dimS*2), spectra_info.SpectroscopyData)
spectra_data = np.array(spectra_data).reshape((2, dimS, dimX, dimY, dimZ))
# Get the activate spectra
spectra_real = spectra_data[0]
spectra_imag = spectra_data[1]
spectra_complex = spectra_real + 1j * spectra_imag
specProc = np.flip(np.flip(np.moveaxis(spectra_complex, [0,1,2,3], [0,2,1,3]),2),3)
I, J, K = np.unravel_index(activated_index, metabolic_map.shape)
num_spectra = len(I)
spectra = np.empty((num_spectra,dimS), dtype=complex)
for m in range(num_spectra):
    spectra[m] =  specProc[:,I[m], J[m], K[m]]

# Split real and imaginary parts
data_real = spectra.real
data_imag = spectra.imag

# Normalize to (-1,1)
max_real = np.array([max(data_real[i]) for i in range(num_spectra)])
max_imag = np.array([max(data_imag[i]) for i in range(num_spectra)])

data_real = np.array([(data_real[i] + max_real[i]) / (2 * max_real[i]) * 2 - 1 for i in range(num_spectra)])
data_imag = np.array([(data_imag[i] + max_imag[i]) / (2 * max_imag[i]) * 2 - 1 for i in range(num_spectra)])

# Set NaN values to 0 
data_real[np.isnan(data_real)] = 0
data_imag[np.isnan(data_imag)] = 0
spectra = np.array([data_real, data_imag])
mat_dict['spectra'] = spectra



io.savemat('/home/kreitnerl/mrs-gan/t10169.mat', mdict=mat_dict)
print('END')