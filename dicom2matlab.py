from options.dicom2matlab_options import Dicom2MatlabOptions
import numpy as np
import scipy.io as io
from util.util import progressbar, is_set_of_type
from util.load_activated_spectra import *
from data.image_folder import make_dataset
from util.process_spectra import preprocess_numpy_spectra

opt = Dicom2MatlabOptions().parse()

def remove_outliers(quantities:dict, num_indices: int):
    valid_indices = np.ones(num_indices, dtype=bool)
    for met_qantities in quantities.values():
        valid_indices &= (met_qantities[-1] < opt.outlier_cutoff) & (met_qantities[-1] != 0)
    return valid_indices

def extract_from_DCM(sourceDir: str, file_name_spec: str, file_ext_metabolite:str, metabolites: list, relativator_met: str, save_dir: str, double_check_activated=None):
    spectra_paths = sorted(make_dataset(sourceDir, file_ext=file_name_spec))
    num_patient = len(spectra_paths)
    print('Number of patients:', num_patient)
    metabolite_paths = {}
    for metabolite in metabolites:
        metabolite_paths[metabolite] = sorted(make_dataset(sourceDir, file_ext=file_ext_metabolite + metabolite + '.dcm'))
    activation_map_paths=sorted(make_dataset(sourceDir, file_ext=file_ext_metabolite + relativator_met + '.dcm'))

    if double_check_activated:
        double_check_map_paths = sorted(make_dataset(sourceDir, file_ext=double_check_activated))
    else:
        double_check_map_paths=None

    assert len(metabolite_paths) == len(metabolites)

    spectra_real = []
    spectra_imag = []
    quantities = {k: [] for k in metabolites}

    num_spectra_per_patient = []
    for i in progressbar(range(num_patient), "Processing patient data: ", 20):
        activated_indices, shape, relativator_values = get_activated_indices(activation_map_paths[i], None if not double_check_activated else double_check_map_paths[i])
        for metabolite, paths in metabolite_paths.items():
            quantities[metabolite].append(get_activated_metabolite_values(paths[i], activated_indices, shape, relativator_values))
        valid_indices = remove_outliers(quantities, len(relativator_values))
        num_spectra_per_patient.append(sum(valid_indices))
        # Sort out invalid voxels for every metabolite
        for metabolite in metabolites:
            quantities[metabolite][-1] = quantities[metabolite][-1][valid_indices]
                    
        dataR, dataI = get_activated_spectra(spectra_paths[i], activated_indices, shape)
        spectra_real.append(dataR[valid_indices])
        spectra_imag.append(dataI[valid_indices])

    spectra_real = np.concatenate(spectra_real, axis=0)
    spectra_imag = np.concatenate(spectra_imag, axis=0)

    for metabolite in metabolites:
        quantities[metabolite] = np.concatenate(quantities[metabolite], axis=0)
    quantities.update({'num_spectra_per_patient': num_spectra_per_patient})

    spectra = preprocess_numpy_spectra(spectra_real, spectra_imag, spectra_real.shape, save_dir, opt)
    spectra = spectra.transpose(1, 2)

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

    spectra, quantities, num_spectra_per_patient = extract_from_DCM(source_dir, opt.file_ext_spectra, opt.file_ext_metabolite, ['cho', 'NAA'], 'cre', opt.save_dir, opt.double_check_activated)
    
    if opt.real:
        name = 'spectra_real'
    elif opt.imag:
        name = 'spectra_imag'
    elif opt.mag:
        name = 'spectra_magnitude'
    else:
        name = 'spectra'
    export_mat({'spectra': np.array(spectra), 'num_spectra_per_patient': num_spectra_per_patient}, opt.save_dir + name)
    export_mat(quantities, save_dir + 'quantities')
    print('Done. You can find the exported matlab file at', save_dir + 'quantities.mat')

dcm2mat(opt.source_dir,  opt.save_dir)