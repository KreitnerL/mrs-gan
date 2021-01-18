import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
import random
from models.auxiliaries.physics_model import PhysicsModel
import torch

# path = '/home/kreitnerl/Datasets/syn_4_real/dataset_spectra.mat'
path_UCSF = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data_corrected.mat'
path_syn_ucsf = '/home/kreitnerl/Datasets/syn_ucsf_corrected/dataset_spectra.mat'
path_quantities = '/home/kreitnerl/Datasets/UCSF_TUM_MRSI/MRSI_data_corrected.mat'

num_plots = 10

var_name = 'spectra'
ppm_range = [7.171825,-0.501875]
crop_range_UCSF=slice(210,722)
crop_range = crop_range_UCSF = slice(300,812)
x = np.linspace(*ppm_range, 1024)[slice(300,812)]
indices = [random.randint(0, 1500) for i in range(10)]

def plotUCSF():
    spectra = io.loadmat(path_UCSF)[var_name]
    # spectra[:,:,623:]=0
    spectra = spectra[:,:,crop_range_UCSF]
    plt.figure()
    for index,i in enumerate(indices):
        spectrum = spectra[i]/np.amax(abs(spectra[i]))
        for j in (0,1):
            plt.plot(x, spectrum[j])
        plt.title('UCSF Spectrum %d'%i)
        plt.xlim([x[0], x[-1]])
        plt.xlabel('ppm')
        # plt.ylim(-1,1)
        plt.savefig('ucsf_ideal/UCSFspectrum%d.png'%index, format='png', bbox_inches='tight')
        plt.cla()

def plotSYN():
    spectra = io.loadmat(path_syn_ucsf)[var_name]
    spectra = spectra[:,:,crop_range]
    plt.figure()
    for index,i in enumerate(indices):
        spectrum = spectra[i]/np.amax(abs(spectra[i]))
        for j in (0,1):
            plt.plot(x, spectrum[j])
        plt.title('Syn_real Spectrum %d'%i)
        plt.xlim([x[0], x[-1]])
        plt.xlabel('ppm')
        # plt.ylim(-1,1)
        plt.savefig('ucsf_ideal/spectrum%d.png'%index, format='png', bbox_inches='tight')
        plt.cla()

def plotIdeal():
    d = io.loadmat(path_quantities)
    cho = np.squeeze(d['cho'])
    naa = np.squeeze(d['naa'])
    torch.cuda.set_device(7)
    opt = Namespace(**{'roi': slice(457,713), 'mag': True, 'ppm_range': ppm_range, 'full_data_length': 1024})
    pm = PhysicsModel(opt).cuda()

    plt.figure()
    for index,i in enumerate(indices):
        params = torch.tensor([[cho[i], naa[i]]]).cuda()/3.6
        spectrum = np.squeeze(pm.forward(params).detach().cpu().numpy())
        plt.cla()
        plt.plot(x,spectrum.transpose())
        plt.title('Ideal Spectrum %d'%i)
        plt.xlabel('ppm')
        plt.xlim([x[0], x[-1]])
        plt.savefig('ucsf_ideal/ideal_spectrum%d.png'%index, format='png', bbox_inches='tight')
    print('Done.')

plotSYN()
plotUCSF()
plotIdeal()
