from argparse import Namespace
from models.auxiliaries.mrs_physics_model import MRSPhysicsModel
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.set_device(7)
ppm_range = [7.171825,-0.501875]
opt = Namespace(**{'roi': slice(300,812), 'mag': True, 'ppm_range': ppm_range, 'full_data_length': 1024})
x = np.linspace(ppm_range, 1024)[opt.roi]
pm = MRSPhysicsModel(opt).cuda()
pm.plot_basisspectra('basis_spec.png', True)

# params = torch.tensor([[2.607970112079701, 1.641095890410959]]).cuda()
# spectrum = np.squeeze(pm.forward(params).detach().cpu().numpy())

# plt.figure()
# plt.plot(x,spectrum.transpose())
# plt.title('Ideal Spectrum')
# plt.xlabel('ppm')
# plt.xlim([x[0], x[-1]])
# plt.savefig('ucsf_ideal/ideal_spectrum.png', format='png', bbox_inches='tight')
# print('Done.')

