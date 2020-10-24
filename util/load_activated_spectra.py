# Convert DCM file to matlab file. Called by preprocess_UCSF.py
# Matlab file will later be loaded by the DataLoader
import pydicom
import numpy as np
import struct

def load_dicom(path):
    print('load', path)
    return pydicom.dcmread(path)

def get_activated_spectra(metabolic_map_path, spectra_path):
    """
    Extracts the activated spectra from a given dicom file.

    Parameters
    ----------
        - metabolicMapPath: file path of the dicom file containing the metabolite information
        - spectraPath: : file path of the dicom file containing the spectra
    
    Returns
    -------
        - Real part of the spectra. Numpy array of shape (N x L), where N is the number of activated spectra and L the length of the spectra
        - Imaginary part of the spectra. Numpy array of shape (N x L), where N is the number of activated spectra and L the length of the spectra
    """
    ############################################################
    # Load the matabolic map to know which voxels are activated
    ############################################################
    metabolic_map_info = load_dicom(metabolic_map_path)
    dimZ = int(metabolic_map_info.NumberOfFrames)
    dimY = int(metabolic_map_info.Columns)
    dimX = int(metabolic_map_info.Rows)
    num_voxels = dimX*dimY*dimZ

    # Stored as a byte array where each value takes 2 bytes!
    data = metabolic_map_info.PixelData
    # Transform to a short array
    data = struct.unpack('h' * num_voxels, data)
    # Most of the voxels are not activated. They contain the a fix baseline value
    baseline = max(data, key=data.count)

    data = np.array(data).reshape((dimX, dimY, dimZ))
    metabolic_map = np.flip(np.flip(data, 1), 2)
    # Get the index of all activated voxels
    activated_index = [i for i, val in enumerate(metabolic_map.flatten()) if val != baseline]

    ############################################################
    # Load the activated spectra
    ############################################################
    spectra_info = load_dicom(spectra_path)
    if 'DataPointColumns' in spectra_info:
        dimS = spectra_info.DataPointColumns
    else:
        dimS = 2048
    # Stored as a byte array where each value takes 4 bytes!
    spectra_data = spectra_info.SpectroscopyData
    # Transform to a float array
    spectra_data = struct.unpack('f' * (num_voxels*dimS*2), spectra_data)
    spectra_data = np.array(spectra_data).reshape((2, dimS, dimX, dimY, dimZ))
    # Get the activate spectra
    spectra_real = spectra_data[0]
    spectra_imag = spectra_data[1]
    spectra_complex = spectra_real + 1j * spectra_imag
    specProc = np.flip(np.flip(np.moveaxis(spectra_complex, [0,1,2,3], [0,2,1,3]),2),3)
    I, J, K = np.unravel_index(activated_index, metabolic_map.shape)
    num_spectra = len(I)
    spectra = np.empty((num_spectra,dimS))
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

    return data_real, data_imag