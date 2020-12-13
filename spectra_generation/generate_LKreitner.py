import argparse
import os

import numpy as np
import scipy.io as io
import torch
# from modules.physics_model.physics_model_physiological import PhysicsModelv3

from physics_model_physiological import PhysicsModelv3
from scipy.interpolate import CubicSpline

# python ./spectra_generation/generate_LKreitner.py --savedir '/home/kreitnerl/Datasets/syn_4_real/dataset' --totalEntries 100000 --not_simple --phase train --pair


def train(path, totalEntries=200000, simple=False, pair=False, blank=None, fixed_params=[]): # path,
    basename = path + 'dataset'
    params = torch.empty((totalEntries, 23)).uniform_(0,1)

    # Metabolite quantities
    print('>>> Metabolite Quantities')
    # params[:,1].fill_(1)
    # cho cre naa glx ins mac lip: t2_min 0.05-->0.0 max:0.6
    params[:,0] = ((3.5 - 0.01) * params[:,0] + 0.01) / 3.5 # / (0.9 - 0.1))
    params[:,1] = 1.0
    params[:,2] = ((3.5 - 0.01) * params[:,2] + 0.01) / 3.5# / (3.8 - 0.05))
    params[:,3] = ((5.1 - 0.05) * params[:,3] + 0.05) / 5.1# / (5.1 - 0.05))
    params[:,4] = ((2.0 - 0.01) * params[:,4] + 0.01) / 2# / (2.0 - 0.01))
    params[:,5] = ((4 - 1) * params[:,5] + 1) / 4# / (4 - 1))
    params[:,6] = ((4 - 1) * params[:,6] + 1) / 4# / (4 - 1))

    for i in range(len(fixed_params)):
        params[:,i] = fixed_params[i]

    # if simple:
    params[:,3].fill_(0)
    params[:,4].fill_(0)
    params[:,5].fill_(0)
    params[:,6].fill_(0)


    for n in range(7):
        if not n==1: # Creatine
            sign = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(params.shape[0])])
            params[sign,int(n)].fill_(0.)
            params[sign,int(n+7)].fill_(0.) # If the lines are omitted, then the broadening is too


    print('>>> Line Broadening')
    for n in range(7,14):
        if simple:
            params[:,n].fill_(1.)
        else:
            # sign = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(params.shape[0])])
            params[:,n] = ((0.6 - 0.002) * params[:,n] + 0.002) / 0.6


    # Frequency Shift - zeroed out
    print('>>> Frequency Shift')
    #
    ind = tuple([14])
    for i in ind:
        params[:,i] = 0.5

        # if simple:
        #     params[:,i] = 0.5
        # else:
        #     sign = torch.tensor([True if torch.rand([1]) > 0.8 else False for _ in range(params.shape[0])])
        #     params[sign,i].fill_(0.5)

    # Phase Shift - set to 0
    print('>>> Phase Shift')
    ind = tuple([16,17])
    for i in ind:
        params[:,i] = 0.5

    # Noise - SNR = 100
    print('>>> Noise')
    # params[:,15] = (80-12.5) * params[:,15].fill_(1) + (12.5 / (80-12.5))
    if simple:
        params[:,15].fill_(1)#.47873799725652)#1)
    else:
        params[:,15].fill_(0.12)
        # params[:,15].fill_(1)

    # Baseline - Entire baseline omitted
    print('>>> Baseline')
    ind = tuple([18,19,20,21,22])
    for i in ind:
        params[:,i].fill_(0)

    crop_range = (1005, 1280)
    model = PhysicsModelv3(cropped=False,magnitude=True, range=crop_range)
    dictionary, index = model.initialize()
    generate_and_save(model, params, simple, path)
    if pair and not simple:
        params[:,3].fill_(0)
        params[:,4].fill_(0)
        params[:,5].fill_(0)
        params[:,6].fill_(0)
        for n in range(7,14):
            params[:,n].fill_(1.)
        params[:,15].fill_(2)
        generate_and_save(model, params, not simple, path+'_ideal')

def generate_and_save(model, params, simple, path):
    print('>>> Generating Spectra')
    spectra, magnitude, parameters = model.forward(params, gen=True, simple=simple)#background_spectra=background)
    quantities = model.quantify(torch.from_numpy(np.expand_dims(params, axis=-1)))
    print('>>> # of spectra: ',spectra.shape[0])
    l = 2048

    target_indices = [430, 590, 680]# [344, 434, 594]
    starting_indices = [1072, 1116, 1192] #[856, 932, 976] #[1072, 1116, 1192] # [536, 558, 596] #

    ratio_start = target_indices[0] / 1024
    ratio_end = target_indices[2] / 1024

    rescale = (target_indices[2] - target_indices[0]) / (starting_indices[2] - starting_indices[0])
    dL_start = target_indices[0]
    L0 = dL_start / rescale
    dL_end = 1024 - target_indices[2]
    L1 = dL_end / rescale
    strt = (starting_indices[0] - L0)
    endd = (starting_indices[2] + L1)
    cropRange = [strt, endd]
    print('Cropping range: ', cropRange)

    delta = cropRange[1] - cropRange[0]
    # spectra =
    n = delta / (1024-1)
    new = np.arange(start=strt, stop=endd+n, step=n)
    # Complex spectra
    xaxis = np.arange(len(spectra[0,0,:]))
    print('>>> Resampling the complex spectra')
    cs_interp = CubicSpline(xaxis, np.asarray(spectra), axis=-1)
    spectra = cs_interp(new)
    # Magnitude spectra
    print('>>> Resampling the magnitude spectra')
    cs_interp = CubicSpline(xaxis, np.asarray(magnitude), axis=-1)
    magnitude = cs_interp(new)

    # spectra[:,0,:] = np.interp(new, np.asarray(models.ppm), spectra[:,0,:])
    # spectra[:,1,:] = np.interp(new, np.asarray(models.ppm), spectra[:,1,:])

    _save(path, torch.from_numpy(spectra), torch.from_numpy(magnitude), parameters, quantities)
    # _save(path, spectra, magnitude, parameters)

def _save(path, spectra, magnitude, parameters, quantities=False):
    print('>>> Saving Spectra')
    base, _ = os.path.split(path)
    os.makedirs(base,exist_ok=True)
    io.savemat(path + '_spectra.mat',do_compression=True,
           mdict={'spectra':np.asarray(spectra)})
    print(path + '_spectra.mat')
    io.savemat(path + '_magnitude.mat',do_compression=True,
           mdict={'mag':np.asarray(magnitude)})
    print(path + '_magnitude.mat')
    io.savemat(path + '_parameters.mat',do_compression=True,
           mdict={'cho':np.asarray(parameters[:,0]),
                  'cre':np.transpose(np.asarray(parameters[:,1])),
                  'naa':np.asarray(parameters[:,2]),
                  'glx':np.asarray(parameters[:,3]),
                  'ins':np.asarray(parameters[:,4]),
                  'mac':np.asarray(parameters[:,5]),
                  'lip':np.asarray(parameters[:,6]),
                  't2':np.asarray(parameters[:,7:14]),
                  'freq_shift':np.asarray(parameters[:,14]),
                  'noise':np.asarray(parameters[:,15]),
                  'phase':np.asarray(parameters[:,(16,17)]),
                  'base_scale0':np.asarray(parameters[:,18]),
                  'base_scale1':np.asarray(parameters[:,19]),
                  'base_scale2':np.asarray(parameters[:,20]),
                  'base_scale3':np.asarray(parameters[:,21]),
                  'base_scale4':np.asarray(parameters[:,22])})
    print(path + '_parameters.mat')
    # if not quantities is None:
    io.savemat(path + '_quantities.mat',do_compression=True,
           mdict={'cho':np.asarray(quantities[:,0]),
                  'cre':np.transpose(np.asarray(quantities[:,1])),
                  'naa':np.asarray(quantities[:,2]),
                  'glx':np.asarray(quantities[:,3]),
                  'ins':np.asarray(quantities[:,4]),
                  'mac':np.asarray(quantities[:,5]),
                  'lip':np.asarray(quantities[:,6]),
                  't2':np.asarray(quantities[:,7:14]),
                  'freq_shift':np.asarray(quantities[:,14]),
                  'noise':np.asarray(quantities[:,15]),
                  'phase':np.asarray(quantities[:,(16,17)]),
                  'base_scale0':np.asarray(quantities[:,18]),
                  'base_scale1':np.asarray(quantities[:,19]),
                  'base_scale2':np.asarray(quantities[:,20]),
                  'base_scale3':np.asarray(quantities[:,21]),
                  'base_scale4':np.asarray(quantities[:,22])})
    print(path + '_quantities.mat')


def quantify(inputdir, savedir):
    parameters = io.loadmat(inputdir)
    keys = ['cho','cre','naa','glx','ins','mac','lip','t2','freq_shift','noise',
            'phase','base_scale0','base_scale1','base_scale2','base_scale3','base_scale4']
    ind = [(0),(1),(2),(3),(4),(5),(6),tuple(np.arange(7,14)),(14),(15),tuple(np.arange(16,18)),(18),(19),(20),(21),(22)]
    compilation = np.zeros([100000, 23])
    for k, i in zip(keys, ind):
        s = parameters[k].shape
        print(k,': ',s)        
        if s[0]<s[1]: 
            if s[0]==1:
                compilation[:,i] = np.transpose(parameters[k])[:,0]
            elif s[0]> 1:
                for ii, n in zip(i,np.arange(0,len(i))):
                    print(ii,n)
                    compilation[:,ii] = np.transpose(parameters[k])[:,n]
        else:
            if s[1]==1:
                compilation[:,i] = parameters[k][:,0]
            elif s[1]> 1:
                for ii, n in zip(i,np.arange(0,len(i))):
                    print(ii,n)
                    compilation[:,ii] = parameters[k][:,n]

    # compilation = np.concatenate(compilation,axis=1)

    comp = np.squeeze(np.asarray(compilation))
    if comp.shape[0]<comp.shape[1]:
        comp = np.transpose(comp)

    model = PhysicsModelv3()
    dictionary, index = model.initialize()

    quantities = model.quantify(torch.from_numpy(comp).unsqueeze(-1))

    os.makedirs(savedir,exist_ok=True)
    io.savemat(os.path.join(savedir, 'dataset_quantities.mat'),do_compression=True,
           mdict={'cho':np.asarray(quantities[:,0]),
                  'cre':np.asarray(quantities[:,1]),
                  'naa':np.asarray(quantities[:,2]),
                  'glx':np.asarray(quantities[:,3]),
                  'ins':np.asarray(quantities[:,4]),
                  'mac':np.asarray(quantities[:,5]),
                  'lip':np.asarray(quantities[:,6]), 
                  't2':np.asarray(quantities[:,7:14]),
                  'freq_shift':np.asarray(quantities[:,14]),
                  'noise':np.asarray(quantities[:,15]),
                  'phase':np.asarray(quantities[:,(16,17)]),
                  'base_scale0':np.asarray(quantities[:,18])*4,
                  'base_scale1':np.asarray(quantities[:,19])*4,
                  'base_scale2':np.asarray(quantities[:,20])*4,
                  'base_scale3':np.asarray(quantities[:,21])*4,
                  'base_scale4':np.asarray(quantities[:,22])*4})

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir',type=str,default='./dataset/dataset')
    parser.add_argument('--totalEntries',type=int,default=None)
    parser.add_argument('--phase',type=str)
    parser.add_argument('--snr',type=tuple,default=(0.80, 0.50, 0.30, 0.10, 0.03, 0.00))
    parser.add_argument('--echo',type=list,default=['short','long'])
    parser.add_argument('--background', action='store_true', default=False, help='add background signals')
    parser.add_argument('--blank', type=float, default=None, help='Percentage of spectra to only contain background')
    parser.add_argument('--inputdir',type=str)
    parser.add_argument('--not_simple',action='store_false',default=True)
    parser.add_argument('--pair',action='store_true', default=False)
    parser.add_argument('--param_path',type=str, default=None)

    args = parser.parse_args()


    path = args.savedir
    if args.phase=='train':
        entries = 200000 if not args.totalEntries else args.totalEntries
        train(totalEntries=entries,path=path,simple=args.not_simple, pair = args.pair)
    elif args.phase=='test':
        print('Needs updating...')
        pass
    elif args.phase=='quantify':
        quantify(inputdir=args.inputdir,savedir=path)
        # entries = 10000 if not args.totalEntries else args.totalEntries
        # test(totalEntries=entries,path=args.savedir,snr=args.snr,echo=args.echo)
    elif args.phase=='pair':
        quantitites = io.loadmat(args.param_path)
        params = []
        params.append(torch.from_numpy(quantitites['cho']/3.5))
        num_samples = params[0].shape[1]
        params.append(torch.ones(num_samples))
        params.append(torch.from_numpy(quantitites['naa']/3.5))
        train(totalEntries=num_samples,path=path,simple=True, pair = False, fixed_params=params)


