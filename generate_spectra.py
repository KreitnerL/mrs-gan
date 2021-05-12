import argparse
import torch
import time
from argparse import Namespace
from models.auxiliaries.mrs_physics_model import MRSPhysicsModel
import scipy.io as io

parser = argparse.ArgumentParser()
parser.add_argument('--quantitity_path', type=str, help='path to the matlab file containing predefined qunatities')
parser.add_argument('--save_path', type=str, help='path where the matlab file with the spectra should be saved')
parser.add_argument('--gpu_id', type=int, default=0, help='Id of the used GPU')
parser.add_argument('--N', type=int, default=100000, help='Number of used spectra')
parser.add_argument('--batch_size', type=int, default=10000, help='Max number of spectra that can be processed at once')
parser.add_argument('--ppm_range', type=str, default='7.171825,-0.501875', help='ppm range of the basis spectra')
parser.add_argument('--crop_range', type=str, default='None,None', help='Region of interest in points where the full spectra has 1024 points')
parser.add_argument('--SNR_min', type=float, default=12, help='Min SNR in dB')
parser.add_argument('--SNR_max', type=float, default=12, help='Max SNR in dB')
parser.add_argument('--β_min', type=float, default=0.08, help='Minimal line-broadening factor in percent. 1 is normal(=min) line broadening')
parser.add_argument('--β_max', type=float, default=0.08, help='Maximum line-broadening factor in percent. 1 is normal(=min) line broadening')
args = parser.parse_args()

args.ppm_range = [*eval(args.ppm_range)]
args.crop_range = slice(*eval(args.crop_range))

opt = Namespace(**{'roi': args.crop_range, 'mag': False, 'ppm_range': args.ppm_range, 'full_data_length': 1024})
torch.cuda.set_device('cuda:'+ str(args.gpu_id))
pm = MRSPhysicsModel(opt).cuda()

if args.quantitity_path:
    quantitites = io.loadmat(args.quantitity_path)
    concentrations = []
    concentrations.append(torch.from_numpy(quantitites['cho'][:,:args.N]/(3.5)))
    concentrations.append(torch.ones((1,args.N), dtype=torch.float64))
    concentrations.append(torch.from_numpy(quantitites['naa'][:,:args.N]/(3.5)))
    concentrations = torch.cat(concentrations, dim=0).transpose(0,1)
    args.N = concentrations.shape[1]
else:
    concentrations = (3.5 - 0.01) * torch.rand(args.N,3) + 0.01

start_time = time.time()
spectra_batches = []
for i in range(int(args.N/args.batch_size) + 1):
    offset = i*args.batch_size
    spectra_batch: "torch.Tensor" = pm.build_spectra(concentrations[offset : min(args.N,offset+args.batch_size)].cuda(), args.β_min, args.β_max, args.SNR_min, args.SNR_max)
    spectra_batches.append(spectra_batch.cpu())
print("--- Generated {0} spectra in {1} seconds ---".format(args.N, time.time() - start_time))

spectra = torch.cat(spectra_batches, dim=0).numpy()
print('Saving spectra...')
io.savemat(args.save_path, mdict = {'spectra': spectra}, do_compression=True)

print('Done. You can find your output at', args.save_path)