import argparse
import os

import numpy as np
import scipy.io as io
import torch
# import torch.nn as nn
from modules.updated_physics_modelv2 import PhysicsModelv2


# class generate(nn.Module):
#     def __init__(self, phase, gpu_ids=[]):
#         super(generate, self).__init__()
#         self.gpu_ids = gpu_ids
#         # if len(gpu_ids)>>0:
#         if phase=='test':
#             self.model = test()
#         elif phase=='train':
#             self.model = train()
#         self.conv = nn.Conv1d(2,62,kernel_size=3)

#     def forward(self, **kwargs):
#         self.model.forward(**kwargs)


def test(path, snr, echo, totalEntries=10000):
    # echo = ['short','long']
    # snr = (0.80, 0.50, 0.30, 0.10, 0.03, 0.00)
      
    for te in echo:
      for s in snr:
        name = te + '_echo_SNR_{}'.format(int(s*100))
        basename = os.path.join(path,'test_sets/' + name)
        try:
            os.makedirs(basename)
        except OSError:
            pass

        params = torch.empty((totalEntries, 23, 1)).uniform_(0,1.0000001)

        # SNR
        print('>>> SNR = {}'.format(s*100))
        params[:,15] = s 

        # Metabolite 
        print('>>> Metabolite Quantities')
        if te=='long':
            sign = torch.tensor([1 if torch.rand([1]) > 0.1 else -1 for _ in range(params.shape[0])]).unsqueeze(-1)
            for n in range(1):
                temp = torch.tensor([1 if torch.rand([1]) > 0.1 else -1 for _ in range(params.shape[0])]).unsqueeze(-1)
                sign = torch.cat((sign, temp), dim=-1)

                params[:,2].fill_(1)
                params[:,(3,10)].fill_(0)
                params[:,(4,11)].fill_(0)

                for i in range(totalEntries):
                    # 
                    if sign[i,0]==1:
                        params[i,(5,12)] = 0
                    else:
                        params[i,5] /= 10
                    if sign[i,1]==1:
                        params[i,(6,13)] = 0
                    else:
                        params[i,6] /= 10
                    # Allow for zeroing out of signals
                    if params[i,0]<<(0.05/0.95).float():
                        params[i,0] = 0.0
                    params[i,1] = 0.0
                    if params[i,2]<<(0.05/3.8).float():
                        params[i,2] = 0.0
                    if params[i,3]<<(0.08/6.5).float():
                        params[i,3] = 0.0
                    # if params[i,4]<<(0.08/6.5).float():
                    if params[i,5]<<0.25:
                        params[i,5] = 0.0
                    if params[i,6]<<0.25:
                        params[i,6] = 0.0

        # Frequency Shift
        print('>>> Frequency Shift')
        sign = torch.tensor([True if torch.rand([1]) > 0.5 else False for _ in range(params.shape[0])])
        ind = tuple([14])
        params[:,ind] /= 2
        params[sign,ind] += 0.5

        # Baseline
        print('>>> Baseline')
        if te=='long':
            ind = tuple([18,19,20,21,22])
            sign = torch.tensor([True if torch.rand([1]) > 0.5 else False for _ in range(params.shape[0])])
            params[:,ind] /= 10 # 20
            params[sign,ind] = 0
           
        model = PhysicsModelv2(cropped=False, magnitude=True)
        dictionary, index = model.initialize()

        print('>>> Generating Spectra')
        spectra, magnitude, parameters = model.forward(params, gen=True)

        print('>>> Saving Spectra')
        print('>>> # of spectra: ',spectra.shape[0])
        _save(basename, spectra, magnitude, parameters)


def train(path, totalEntries=200000): # path, 
    basename = path + 'dataset'
    params = torch.empty((totalEntries, 23)).uniform_(0,1)

    # Metabolite quantities
    print('>>> Metabolite Quantities')
    # params[:,1].fill_(1)
    # cho cre naa glx ins mac lip: t2_min 0.05-->0.0 max:0.6
    params[:,0] = (1.1 - 0.1) * params[:,0] + (0.1 / (1.1 - 0.1))
    params[:,1] = 1.0
    params[:,2] = (4 - 0.05) * params[:,2] + (0.05 / (4 - 0.05))
    params[:,3] = (6.5 - 0.05) * params[:,3] + (0.05 / (6.5 - 0.05))
    params[:,4] = (2.7 - 0.01) * params[:,4] + (0.01 / (2.7 - 0.01))
    params[:,5] = (4 - 1) * params[:,5] + (1 / (4 - 1))
    params[:,6] = (4 - 1) * params[:,6] + (1 / (4 - 1))
    for n in range(7,13):
        params[:,n] = (0.6 - 0.002) * params[:,n] + (0.002 / (0.6 - 0.002))
    for n in range(4):
        sign = torch.tensor([True if torch.rand([1]) > 0.5 else False for _ in range(params.shape[0])])
        params[sign,int(n+3)].fill_(0.)
        params[sign,int(n+10)].fill_(0.) # If the lines are omitted, then the broadening is too

    # Frequency Shift
    print('>>> Frequency Shift')
    # sign = torch.tensor([True if torch.rand([1]) > 0.5 else False for _ in range(params.shape[0])])
    # ind = tuple([14])#,15,16,17,18,19,20])
    # params[:,ind] /= 2
    # for i in ind:
    #     params[sign,i] += 0.5

    # Noise
    params[:,15] = (.80-.20) * params[:,15] + (0.2 / (.80-.20))

    # Baseline
    print('>>> Baseline')
    sign = torch.tensor([True if torch.rand([1]) > 0.5 else False for _ in range(params.shape[0])])
    ind = tuple([18,19,20,21,22])
    for i in ind:
        params[sign,i].fill_(0)

    model = PhysicsModelv2(cropped=False,magnitude=True)
    dictionary, index = model.initialize()


    print('>>> Generating Spectra')
    spectra, magnitude, parameters = model.forward(params, gen=True)
    print('>>> # of spectra: ',spectra.shape[0])
    _save(path, spectra, magnitude, parameters)



def _save(path, spectra, magnitude, parameters):
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
                  'base_scale4':np.asarray(parameters[:,22]),
                  'crop':np.asarray([1005,1325])})
    print(path + '_parameters.mat')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir',type=str,default='./dataset/dataset')
    parser.add_argument('--totalEntries',type=int,default=None)
    parser.add_argument('--phase',type=str)
    parser.add_argument('--snr',type=tuple,default=(0.80, 0.50, 0.30, 0.10, 0.03, 0.00))
    parser.add_argument('--echo',type=list,default=['short','long'])
    # parser.add_argument('--G0', action='append_const', dest='gpu_ids', const=0, help='use GPU 0 in addition to the default GPU')
    # parser.add_argument('--G1', action='append_const', dest='gpu_ids', const=1, help='use GPU 1 in addition to the default GPU')
    # parser.add_argument('--G2', action='append_const', dest='gpu_ids', const=2, help='use GPU 2 in addition to the default GPU')

    args = parser.parse_args()

    # if args.gpu_ids:
    #     length = len(args.gpu_ids)
    #     if length == 1:
    #         args.gpu_ids = [int(args.gpu_ids[0])]
    #     elif length == 2:
    #         args.gpu_ids = [int(args.gpu_ids[0]), int(args.gpu_ids[1])]
    #     elif length == 3:
    #         args.gpu_ids = [int(args.gpu_ids[0]), int(args.gpu_ids[1]), int(args.gpu_ids[2])]
    #     assert torch.cuda.is_available()
    #     torch.cuda.device(args.gpu_ids[0])
    #     torch.cuda.init()
    # else:
    #     args.gpu_ids = []

    # model = generate(phase=args.phase)
    # if len(args.gpu_ids) >> 0:
    #     model.to(args.gpu_ids[0])
    # global path
    path = args.savedir
    if args.phase=='train':
        entries = 200000 if not args.totalEntries else args.totalEntries
        # print(args.savedir, len(args.savedir))
        # print(args.savedir[0], len(args.savedir[0]))
        train(totalEntries=entries,path=path)
    elif args.phase=='test':
        entries = 10000 if not args.totalEntries else args.totalEntries       
        test(totalEntries=entries,path=args.savedir,snr=args.snr,echo=args.echo)
