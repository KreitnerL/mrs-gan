from options.dicom2matlab_options import Dicom2MatlabOptions
import numpy as np
import scipy.io as io
from util.util import progressbar, is_set_of_type
from util.load_activated_spectra import *
from data.image_folder import make_dataset
opt = Dicom2MatlabOptions().parse()

def extract_from_DCM(sourceDir: str, file_name_spec: str, file_ext_metabolite:str, metabolites: list):
    spectra_paths = sorted(make_dataset(sourceDir, file_ext=file_name_spec))
    num_patient = len(spectra_paths)
    print('Number of patients:', num_patient)
    metabolite_paths = {}
    for metabolite in metabolites:
        metabolite_paths[metabolite] = sorted(make_dataset(sourceDir, file_ext=file_ext_metabolite + metabolite + '.dcm'))
    
    spectra_real = []
    spectra_imag = []
    quantities = {k: [] for k in metabolites}

    num_spectra_per_patient = []
    for i in progressbar(range(num_patient), "Processing patient data: ", 20):
        num_spectra=None
        shape=None
        activated_indices=None
        for metabolite, paths in metabolite_paths.items():
            met_vals, met_map, shape = load_metabolic_map(paths[i])
            baseline = np.bincount(met_map).argmax()
            # assert baseline != 0
            activated_indices = met_map!=baseline
            met_vals = met_vals[activated_indices]
            quantities[metabolite].append(met_vals)
            num_spectra=len(met_vals)

        num_spectra_per_patient.append(num_spectra)
        dataR, dataI = get_activated_spectra(spectra_paths[i], activated_indices, shape)
        spectra_real.append(dataR)
        spectra_imag.append(dataI)

    for metabolite in metabolites:
        quantities[metabolite] = np.concatenate(quantities[metabolite], axis=0)
    spectra_real = np.expand_dims(np.concatenate(spectra_real, axis=0), axis=-1)
    spectra_imag = np.expand_dims(np.concatenate(spectra_imag, axis=0), axis=-1)
    spectra = np.concatenate([spectra_real, spectra_imag], axis=-1)
    spectra = np.transpose(spectra, (0,2,1))

    return spectra, quantities, num_spectra_per_patient

def export_mat(mat_dict, path):
    """ Saves the given dictionary as a matlab file at the given path"""
    mat_dict = {k.lower() :v for k,v in mat_dict.items()}
    io.savemat(path + '.mat', mdict=mat_dict)

def dcm2mat(source_dir, save_dir):
    """
    This function loads all DICOM files from all subfolders at the given path and stores all spectra and quantities in two seperate matlab files.
    The variable holding the spectra is called 'spectra'.
    The data is transformed like this:
    """
    if not is_set_of_type(source_dir, '.dcm'):
        raise ValueError("Source directory does not contain any valid spectra")

    spectra, quantities, num_spectra_per_patient = extract_from_DCM(source_dir, opt.file_ext_spectra, opt.file_ext_metabolite, ['cre', 'cho', 'NAA'])
    quantities.update({'num_spectra_per_patient': num_spectra_per_patient, 'spectra': np.array(spectra)})
    export_mat(quantities, save_dir + 'quantities_raw')
    print('Done. You can find the exported matlab file at', save_dir + 'quantities_raw.mat')

dcm2mat(opt.source_dir,  opt.save_dir)