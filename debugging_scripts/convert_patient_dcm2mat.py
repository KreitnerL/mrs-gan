import pydicom
import numpy as np
import scipy.io as io
import struct

basepath = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/batch_1/tar_bundle/data/p41/jasonc/BjoernRandomTrees/data_to_share/for_tum_t8771/ucsf/t8771_UCSF_'
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
shape = (dimX,dimY,dimZ)

# Following 3 steps are equal to matlabs "squeeze(double(dicomread(info)))""
# Stored as a byte array where each value takes 2 bytes! 
data = struct.unpack('H' * (num_voxels), metabolic_map_info.PixelData)
data = np.reshape(data, (dimX,dimY,dimZ), order='C')
data = np.moveaxis(data, [0,1,2], [2,0,1])


metabolic_map = np.flip(np.flip(data, 0), 1)
metabolic_map_flat = metabolic_map.flatten('F')

# Most of the voxels are not activated. They contain the a fix baseline value
baseline = np.bincount(metabolic_map_flat).argmax()
# Get the index of all activated voxels
activated_index = (metabolic_map_flat != baseline) & (metabolic_map_flat != 0)

mat_dict={}
for metabolite in metabolites:
    metabolic_map_info = pydicom.dcmread(basepath+ metabolite + '.dcm')
    data = struct.unpack('H' * (num_voxels), metabolic_map_info.PixelData)
    data = np.reshape(data, (dimX,dimY,dimZ), order='C')
    data = np.moveaxis(data, [0,1,2], [2,0,1])
    meta_map = np.flip(np.flip(data, 0), 1).flatten('F')
    mat_dict[metabolite] = meta_map[activated_index]

spectra_info = pydicom.dcmread(basepath+ 'proc.dcm')
if 'DataPointColumns' in spectra_info:
    dimS = spectra_info.DataPointColumns
else:
    dimS = 2048
# The following steps are equal to matlabs "reshape(info.SpectroscopyData,2,dimS,dimX,dimY,dimZ)"
# Stored as a byte array where each value takes 4 bytes!
spectra_data = struct.unpack('f' * (num_voxels*dimS*2), spectra_info.SpectroscopyData)
spectra_data = np.array(spectra_data).reshape((dimZ, dimY, dimX, dimS, 2), order='C')
spectra_data = np.moveaxis(spectra_data, [0,1,2,3,4], [4,3,2,1,0])

# Get the activate spectra
spectra_real = spectra_data[0]
spectra_imag = spectra_data[1]
spectra_complex = spectra_real + 1j * spectra_imag
specProc = np.flip(np.flip(np.moveaxis(spectra_complex, [0,1,2,3], [0,2,1,3]),1),2)
spectra = specProc.reshape((dimS, np.prod(shape)), order='F')[:,activated_index].transpose()
num_spectra = len(spectra)
# Split real and imaginary parts
data_real = spectra.real
data_imag = spectra.imag

# Normalize to (-1,1)
max_real = np.array([max(data_real[i]) for i in range(num_spectra)])
max_imag = np.array([max(data_imag[i]) for i in range(num_spectra)])

data_real = np.array([(data_real[i] + max_real[i]) / (2 * max_real[i]) * 2 - 1 for i in range(num_spectra)])
data_imag = np.array([(data_imag[i] + max_imag[i]) / (2 * max_imag[i]) * 2 - 1 for i in range(num_spectra)])

assert sum(sum(np.isnan(data_real))) == 0
assert sum(sum(np.isnan(data_real))) == 0
# Set NaN values to 0 
data_real[np.isnan(data_real)] = 0
data_imag[np.isnan(data_imag)] = 0
spectra = np.array([data_real, data_imag])
mat_dict['spectra'] = spectra



io.savemat('/home/kreitnerl/mrs-gan/t10169.mat', mdict=mat_dict)
print('END')