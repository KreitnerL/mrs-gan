# Convert DCM file to matlab file. Called by preprocess_UCSF.py
# Matlab file will later be loaded by the DataLoader
import pydicom
import numpy as np
import struct

def load_dicom(path):
    # print('load', path)
    return pydicom.dcmread(path)

def get_activated_indices(metabolic_map_path):
    """
    Indetifies the actiated voxels and returns their indices

    Parameters
    ----------
        - metabolicMapPath: file path of the dicom file containing the metabolite information

    Returns
    -------
        - activated_index: List of indices of activated voxels (Flattened)
        - shape: The shape of the voxel matrix
    """
    ############################################################
    # Load the matabolic map to know which voxels are activated
    ############################################################
    metabolic_map_info = load_dicom(metabolic_map_path)
    dimZ = int(metabolic_map_info.NumberOfFrames)
    dimY = int(metabolic_map_info.Columns)
    dimX = int(metabolic_map_info.Rows)
    shape = (dimX, dimY, dimZ)
    num_voxels = dimX*dimY*dimZ

    # Stored as a byte array where each value takes 2 bytes!
    data = metabolic_map_info.PixelData
    # Transform to a uint16 array
    data = struct.unpack('H' * (num_voxels), data)
    data = np.array(data).reshape(shape, order='F')

    
    metabolic_map = np.flip(np.flip(data, 0), 1)
    metabolic_map_flat = metabolic_map.flatten()
    # Most of the voxels are not activated. They contain the a fix baseline value
    baseline = np.bincount(metabolic_map_flat).argmax()
    # Get the index of all activated voxels
    activated_indices = [i for i, val in enumerate(metabolic_map_flat) if val != baseline and val!=0]
    relativator_values = np.array([val for i,val in enumerate(metabolic_map_flat) if i in activated_indices])
    return activated_indices, shape, relativator_values


def get_activated_spectra(spectra_path, activated_index, shape):
    """
    Extracts the activated spectra from a given dicom file.

    Parameters
    ----------
        - spectraPath: file path of the dicom file containing the spectra
        - activated_index: List of indices of activated voxels (Flattened)
        - shape: The shape of the voxel matrix
    
    Returns
    -------
        - Real part of the spectra. Numpy array of shape (N x L), where N is the number of activated spectra and L the length of the spectra
        - Imaginary part of the spectra. Numpy array of shape (N x L), where N is the number of activated spectra and L the length of the spectra
    """
    ############################################################
    # Load the activated spectra
    ############################################################
    spectra_info = load_dicom(spectra_path)
    num_voxels = np.prod(shape)
    if 'DataPointColumns' in spectra_info:
        dimS = spectra_info.DataPointColumns
    else:
        dimS = 2048

    # Stored as a byte array where each value takes 4 bytes!
    spectra_data = spectra_info.SpectroscopyData
    # Transform to a float array
    spectra_data = struct.unpack('f' * (num_voxels*dimS*2), spectra_data)
    spectra_data = np.array(spectra_data).reshape((2, dimS, *shape), order='F')
    # Get the activate spectra
    spectra_real = spectra_data[0]
    spectra_imag = spectra_data[1]
    spectra_complex = spectra_real + 1j * spectra_imag
    specProc = np.flip(np.flip(np.moveaxis(spectra_complex, [0,1,2,3], [0,2,1,3]),1),2)
    I, J, K = np.unravel_index(activated_index, shape)
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

    assert not any(v == 0 for v in max_real)
    assert not any(v == 0 for v in max_imag)

    data_real = np.array([(data_real[i] + max_real[i]) / (2 * max_real[i]) * 2 - 1 for i in range(num_spectra)])
    data_imag = np.array([(data_imag[i] + max_imag[i]) / (2 * max_imag[i]) * 2 - 1 for i in range(num_spectra)])


    # Set NaN values to 0 
    data_real[np.isnan(data_real)] = 0
    data_imag[np.isnan(data_imag)] = 0

    return data_real, data_imag

def get_activated_metabolite_values(metabolic_map_path, activated_index, shape, relativator_values):
    """
    Extracts the activated metabolite quantities from a given dicom file.

    Parameters
    ----------
        - metabolic_map_path: file path of the dicom file containing the metabolic info
        - activated_index: List of indices of activated voxels (Flattened)
        - shape: The shape of the voxel matrix
    
    Returns
    -------
        - nparray of quantities for activated voxels
    """
    metabolic_map_info = load_dicom(metabolic_map_path)

    # Stored as a byte array where each value takes 2 bytes!
    data = metabolic_map_info.PixelData
    # Transform to a uint16 array
    data = struct.unpack('H' * (np.prod(shape)), data)
    data = np.array(data).reshape(shape, order='F')
    
    metabolic_map = np.flip(np.flip(data, 1), 2)
    metabolic_map_flat = metabolic_map.flatten()
    absolute_quantities = np.array([val for i,val in enumerate(metabolic_map_flat) if i in activated_index])
    relative_quantities = absolute_quantities/relativator_values

    return relative_quantities