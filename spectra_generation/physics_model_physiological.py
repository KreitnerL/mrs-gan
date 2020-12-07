from collections import OrderedDict
import scipy.io as io
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch import fft
import numpy as np

# from collections import OrderedDict

__all__ = ['PhysicsModelv3'] #


def convertdict(file, simple=False, device='cpu'):
    if simple:
        p = SimpleNamespace(**file) # self.basisSpectra
        keys = [y for y in dir(p) if not y.startswith('__')]
        for i in range(len(keys)):
            file[keys[i]] = torch.FloatTensor(np.asarray(file[keys[i]], dtype=np.float64)).squeeze().to(device)
        return SimpleNamespace(**file)
    else:
        delete = []
        for k, v in file.items():
            if not k.startswith('__'):
                file[k] = torch.FloatTensor(np.asarray(file[k], dtype=np.float64)).squeeze().to(device)
            else:
                delete.append(k)
        if len(delete)>0:
            for k in delete:
                file.pop(k, None)
        return file




class PhysicsModelv3(nn.Module):      # Updated 06.24.2020 JTL
    __version__ = ['3']
    def __init__(self, cropped=True, magnitude=True, range=(1000,1250), gpu_ids=[]):
        super(PhysicsModelv3, self).__init__()
        self.gpu_ids = gpu_ids
        # Load basis spectra, concentration ranges, and units
        paths = ['/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/Para_for_spectra_gen.mat',
                 '/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/conc_ranges_for_Linus.mat',
                 '/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/units.mat']

        for path in paths:
            with open(path, 'rb') as file:
                dict = convertdict(io.loadmat(file))
                for key, value in dict.items():
                    self.register_buffer(str(key), value)

        # Rephasing
        self.register_buffer('l', torch.FloatTensor([2048]).squeeze())
        self.register_buffer('j', torch.linspace(1,int(self.l),int(self.l)))

        self.t = self.t.unsqueeze(-1)
        self.cropped = cropped
        self.cropRange = range
        self.mag = magnitude


    def __repr__(self):
        return 'MRS_PhysicsModel(basis={}, lines={}, cropped={}, magnitude={})'.format('ISMRM', 3, self.cropped, self.mag)

    # def __getattr__(self, item):
    #     if item=='ppm':
    #         return self.ppm
    @property
    def _ppm(self):
        return self.ppm

    def initialize(self, metab=['Cho','Cre','Naa','Glx','Ins','Mac','Lip']):
        metab = sorted(metab)
        n = len(metab)
        l = 2 * n + 9
        dict, ind = OrderedDict({}), []
        for i in range(n):
            dict.update({metab[i]: []})
            ind.append((i))                                  # Individual metabolites
        broad = torch.arange(n,2*n)
        ind.append(tuple([int(y) for y in broad]))           # Line broadening values
        ind.append(int(2*n))                                 # Frequency shift
        ind.append(int(2*n+1))                               # Noise
        phase = [2*n+2,2*n+3]
        base = torch.arange(2*n+4,2*n+4 + 5)
        cum_metab = torch.arange(0,n)
        cum_param = torch.arange(n,l)
        cum_total = torch.arange(0,l)
        ind.append(tuple([int(y) for y in phase]))           # Phase correction
        ind.append(tuple([int(y) for y in base]))            # Baselines
        ind.append(tuple([int(y) for y in cum_metab]))       # Cummulative metabolites
        ind.append(tuple([int(y) for y in cum_param]))       # Cummulative parameters
        ind.append(tuple([int(y) for y in cum_total]))       # Cummulative total

        dict.update({'T2': [], 'F_Shift': [], 'Noise': [], 'Phase': [], 'Baseline': [],
                     'Metabolites': [], 'Parameters': [], 'Overall': []})
        return dict, ind

    def quantify(self, params):
        params = params.clone().detach()
        Cho = self.pch_max * params[:,0,:]
        Cre = self.cr_range * params[:,1,:]
        Naa = self.naa_max * params[:,2,:]
        Glx = self.glx_max * params[:,3,:]
        Ins = self.ins_max * params[:,4,:]
        Mac = self.mac_max * params[:,5,:]
        Lip = self.lip_max * params[:,6,:]
        # Line broadening
        chlb = self.t2Vector_max * params[:,7,:]
        crlb = self.t2Vector_max * params[:,8,:]
        nalb = self.t2Vector_max * params[:,9,:]
        gllb = self.t2Vector_max * params[:,10,:]
        inlb = self.t2Vector_max * params[:,11,:]
        malb = self.t2Vector_max * params[:,12,:]
        lilb = self.t2Vector_max * params[:,13,:]
        # Frequency
        fshift = (self.fshift_max - self.fshift_min) * params[:,14,:] + self.fshift_min
        # Noise
        noise = self.snr_max * params[:,15,:]
        # Phase
        phi0 = (self.phi0_max - self.phi0_min) * params[:,16,:] + self.phi0_min
        phi1 = (self.phi1_max - self.phi1_min) * params[:,17,:] + self.phi1_min
        # Cho = (self.pch_max - self.pch_min) * params[:,0,:] + self.pch_min
        # Cre = self.cr_range * params[:,1,:]
        # Naa = (self.naa_max - self.naa_min) * params[:,2,:] + self.naa_min
        # Glx = (self.glx_max - self.glx_min) * params[:,3,:] + self.glx_min
        # Ins = (self.ins_max - self.ins_min) * params[:,4,:] + self.ins_min
        # Mac = (self.mac_max - self.mac_min) * params[:,5,:] + self.mac_min
        # Lip = (self.lip_max - self.lip_min) * params[:,6,:] + self.lip_min
        # # Line broadening
        # chlb = (self.t2Vector_max - self.t2Vector_min) * params[:,7,:] + self.t2Vector_min
        # crlb = (self.t2Vector_max - self.t2Vector_min) * params[:,8,:] + self.t2Vector_min
        # nalb = (self.t2Vector_max - self.t2Vector_min) * params[:,9,:] + self.t2Vector_min
        # gllb = (self.t2Vector_max - self.t2Vector_min) * params[:,10,:] + self.t2Vector_min
        # inlb = (self.t2Vector_max - self.t2Vector_min) * params[:,11,:] + self.t2Vector_min
        # malb = (self.t2Vector_max - self.t2Vector_min) * params[:,12,:] + self.t2Vector_min
        # lilb = (self.t2Vector_max - self.t2Vector_min) * params[:,13,:] + self.t2Vector_min
        # # Frequency
        # fshift = (self.fshift_max - self.fshift_min) * params[:,14,:] + self.fshift_min
        # # Noise
        # noise = (self.snr_max - self.snr_min) * params[:,15,:] + self.snr_min
        # # Phase
        # phi0 = (self.phi0_max - self.phi0_min) * params[:,16,:] + self.phi0_min
        # phi1 = (self.phi1_max - self.phi1_min) * params[:,17,:] + self.phi1_min
        # Baseline
        # no quantification
        params[:,0,:], params[:,1,:], params[:,2,:], params[:,3,:], params[:,4,:], params[:,5,:], params[:,6,:] = Cho, Cre, Naa, Glx, Ins, Mac, Lip
        params[:,7,:], params[:,8,:], params[:,9,:], params[:,10,:], params[:,11,:], params[:,12,:], params[:,13,:] = chlb, crlb, nalb, gllb, inlb, malb, lilb
        params[:,14,:], params[:,15,:], params[:,16,:], params[:,17,:] = fshift, noise, phi0, phi1
        return params

    def rephase(self, signal, phi0, phi1):
        theta = (self.phi0_max - self.phi0_min) * phi0 + self.phi0_min + \
                ((self.phi1_max - self.phi1_min) * phi1 + self.phi0_min) * ((self.j-1) / self.l)
        return complex_adjustment(signal, theta)

    def dephase(self, signal, phi0, phi1):
        return self.rephase(signal, torch.tensor(1-phi0, requires_grad=phi0.requires_grad), torch.tensor(1-phi1, requires_grad=phi1.requires_grad))

    def macroline(self, params):
        mac   = complex_adjustment(self.Mac*self.fidMac,((self.fshift_max - self.fshift_min) * params[:,14,:].squeeze() + self.fshift_min) * self.t)
        mac  *= torch.exp(-self.t/params[:,12,:].squeeze()).t()
        mac   = fftshift(fft(mac.transpose(2,1),1).transpose(2,1))
        theta = (self.phi0_max - self.phi0_min) * params[:,16,:].squeeze() + self.phi0_min + \
                ((self.phi1_max - self.phi1_min) * params[:,17,:].squeeze() + self.phi0_min) * ((self.j-1) / self.l)
        return complex_adjustment(mac, theta)

    def lipidline(self, params):
        lip   = complex_adjustment(self.Lip*self.fidLip,((self.fshift_max - self.fshift_min) * params[:,14,:].squeeze() + self.fshift_min) * self.t)
        lip  *= torch.exp(-self.t/params[:,13,:].squeeze()).t()
        lip   = fftshift(fft(lip.transpose(2,1),1).transpose(2,1))
        theta = (self.phi0_max - self.phi0_min) * params[:,16,:].squeeze() + self.phi0_min + \
                ((self.phi1_max - self.phi1_min) * params[:,17,:].squeeze() + self.phi0_min) * ((self.j-1) / self.l)
        return complex_adjustment(lip, theta)

    @staticmethod
    def broaden(params, index, tmin, tmax, t):
        t2 = torch.empty((params.shape[0],len(index)))
        pair = tuple(torch.arange(0,len(index)))
        for i, v in zip(pair, index):
            t2[:,i] = params[:,v].squeeze()
        t2 *= (tmax - tmin)
        t2 += tmin
        out = (-t.unsqueeze(0).repeat(params.shape[0],1,1) / t2.unsqueeze(1)).permute(0,2,1)
        return out.repeat_interleave(2,dim=1)

    def forward(self, params, gen=False, noise=True, simple=False):
        if params.shape[-1]==1: params = params.squeeze()
        if params.dim()==1: params = params.unsqueeze(0)

        # Define basis spectra coefficients
        if gen: print('>>>>> Preparing metabolite coefficients')
        Cho = torch.unsqueeze(self.pch_max  * params[:,0],1).unsqueeze(-1).repeat(1,2,1)
        Cre = torch.unsqueeze(self.cr_range * params[:,1],1).unsqueeze(-1).repeat(1,2,1)
        Naa = torch.unsqueeze(self.naa_max  * params[:,2],1).unsqueeze(-1).repeat(1,2,1)
        Glx = torch.unsqueeze(self.glx_max  * params[:,3],1).unsqueeze(-1).repeat(1,2,1)
        Ins = torch.unsqueeze(self.ins_max  * params[:,4],1).unsqueeze(-1).repeat(1,2,1)
        Mac = torch.unsqueeze(self.mac_max  * params[:,5],1).unsqueeze(-1).repeat(1,2,1)
        Lip = torch.unsqueeze(self.lip_max  * params[:,6],1).unsqueeze(-1).repeat(1,2,1)


        # Line Broadening
        if gen: print('>>>>> Broadening line shapes')
        if simple:
            fidSum = Cho * self.fidCh  + \
                     Cre * self.fidCr  + \
                     Naa * self.fidNaa + \
                     Glx * self.fidGlx + \
                     Ins * self.fidIns + \
                     Mac * self.fidMac + \
                     Lip * self.fidLip
        else:
            broadening = self.broaden(params, (7,8,9,10,11,12,13), self.t2Vector_min, self.t2Vector_max, self.t)
            fidSum = Cho * self.fidCh  * torch.exp(broadening[:, 0:2, :].squeeze()) + \
                     Cre * self.fidCr  * torch.exp(broadening[:, 2:4, :].squeeze()) + \
                     Naa * self.fidNaa * torch.exp(broadening[:, 4:6, :].squeeze()) + \
                     Glx * self.fidGlx * torch.exp(broadening[:, 6:8, :].squeeze()) + \
                     Ins * self.fidIns * torch.exp(broadening[:, 8:10,:].squeeze()) + \
                     Mac * self.fidMac * torch.exp(broadening[:,10:12,:].squeeze()) + \
                     Lip * self.fidLip * torch.exp(broadening[:,12:14,:].squeeze())


        # Add Baseline - non linear to give wavy effect
        # if gen: print('>>>>> Adding baselines')
        # baseline = torch.unsqueeze(params[:,18],1).unsqueeze(-1).repeat(1,2,1) * self.base1 + \
        #            torch.unsqueeze(params[:,19],1).unsqueeze(-1).repeat(1,2,1) * self.base2 + \
        #            torch.unsqueeze(params[:,20],1).unsqueeze(-1).repeat(1,2,1) * self.base3 + \
        #            torch.unsqueeze(params[:,21],1).unsqueeze(-1).repeat(1,2,1) * self.base4 + \
        #            torch.unsqueeze(params[:,22],1).unsqueeze(-1).repeat(1,2,1) * self.base5
        # fidSum += baseline


        # if background_spectra:
        #     Ace  = torch.unsqueeze(((self.ace_max  - self.ace_min)  * params[:,23] + self.ace_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Ala  = torch.unsqueeze(((self.ala_max  - self.ala_min)  * params[:,24] + self.ala_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Asc  = torch.unsqueeze(((self.asc_max  - self.asc_min)  * params[:,25] + self.asc_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Asp  = torch.unsqueeze(((self.asp_max  - self.asp_min)  * params[:,26] + self.asp_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Gaba = torch.unsqueeze(((self.gaba_max - self.gaba_min) * params[:,27] + self.gaba_min),1).unsqueeze(-1).repeat(1,2,1)
        #     Glc  = torch.unsqueeze(((self.glc_max  - self.glc_min)  * params[:,28] + self.glc_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Gly  = torch.unsqueeze(((self.gly_max  - self.gly_min)  * params[:,29] + self.gly_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Gpc  = torch.unsqueeze(((self.gpc_max  - self.gpc_min)  * params[:,30] + self.gpc_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Gsh  = torch.unsqueeze(((self.gsh_max  - self.gsh_min)  * params[:,31] + self.gsh_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     Pe   = torch.unsqueeze(((self.pe_max   - self.pe_min)   * params[:,32] + self.pe_min)  ,1).unsqueeze(-1).repeat(1,2,1)
        #     Tau  = torch.unsqueeze(((self.tau_max  - self.tau_min)  * params[:,33] + self.tau_min) ,1).unsqueeze(-1).repeat(1,2,1)
        #     broad_back = self.broaden(params, (34,35,36,37,38,39,40,41,42,43,44),self.t2Vector_min, self.t2Vector_max, self.t)
        #     background = Ace  * self.fidAce  * torch.exp(broad_back[:, 0:2, :]) + \
        #                  Ala  * self.fidAla  * torch.exp(broad_back[:, 2:4, :]) + \
        #                  Asc  * self.fidAsc  * torch.exp(broad_back[:, 4:6, :]) + \
        #                  Asp  * self.fidAsp  * torch.exp(broad_back[:, 6:7, :]) + \
        #                  Gaba * self.fidGaba * torch.exp(broad_back[:, 8:10,:]) + \
        #                  Glc  * self.fidGlc  * torch.exp(broad_back[:,10:12,:]) + \
        #                  Gly  * self.fidGly  * torch.exp(broad_back[:,12:14,:]) + \
        #                  Gpc  * self.fidGpc  * torch.exp(broad_back[:,14:16,:]) + \
        #                  Gsh  * self.fidGsh  * torch.exp(broad_back[:,16:18,:]) + \
        #                  Pe   * self.fidPe   * torch.exp(broad_back[:,18:20,:]) + \
        #                  Tau  * self.fidTau  * torch.exp(broad_back[:,20:22,:])
        #     fidSum += background


        # Frequency Shift
        # if gen: print('>>>>> Shifting frequencies')
        # # For no shift, coeff should approach 0.5 - symmetric range!
        # delta, min = self.fshift_max - self.fshift_min, self.fshift_min
        # theta = torch.FloatTensor((delta * params[:,14].unsqueeze(-1) + min).unsqueeze(-1) * self.t.t())
        # fidSum = complex_adjustment(fidSum, theta)

        # Add Noise
        if gen: print('>>>>> Adding noise')
        if simple:
            fidSumNoise = fidSum
        else:
        # # Modeling SNR, not noise. SNR coeff should approach 1 for no noise
            snr_min, snr_max = self.snr_min, self.snr_max # 0, 100
            E = ((snr_max - snr_min) * params[:,15] + snr_min).unsqueeze(-1).unsqueeze(-1)
            snr = torch.max(fidSum.abs(), dim=-1, keepdim=True).values / 10.0**(E.float()/10.0)
            e = torch.distributions.normal.Normal(0,torch.sqrt(2.0*snr.squeeze())/2.0).sample([fidSum.shape[2]])
            if e.dim()==2: e = e.unsqueeze(1).permute(1,2,0)
            elif e.dim()==3: e = e.permute(1,2,0)
            fidSumNoise = fidSum.float() + snr*e if noise else fidSum.float()
        # fidSumNoise = fidSum

        # Recover Spectrum
        if gen: print('>>>>> Recovering spectra')
        specSummed = fftshift(fft(fidSumNoise.transpose(2,1),1).transpose(2,1),-1)

        # Rephasing Spectrum
        # if gen: print('>>>>> Rephasing spectra')
        # theta_a = ((self.phi0_max - self.phi0_min) * params[:,16] + self.phi0_min).unsqueeze(-1)
        # theta_b = ((self.phi1_max - self.phi1_min) * params[:,17] + self.phi0_min).unsqueeze(-1) * ((self.j-1) / self.l).unsqueeze(0)
        # theta = theta_a + theta_b
        # specSummed = complex_adjustment(specSummed, theta.unsqueeze(1))

        ## Normalize Spectra by dividing by the norm of the area under the 3 major peaks
        if gen: print('>>>>> Normalizing spectra')
        print('specSummed.shape: ',specSummed.shape)
        channel_max = torch.max(torch.abs(specSummed),dim=-1).values
        print('channel_max.shape: ',channel_max.shape)
        sample_max = torch.max(torch.abs(channel_max),dim=-1).values
        print('sample_max.shape: ',sample_max.shape)
        denom = sample_max.unsqueeze(-1).unsqueeze(-1).repeat_interleave(2,dim=1)
        print('denom.shape: ',denom.shape)
        spec_norm = specSummed / denom
        # spec_norm = specSummed / torch.max(torch.abs(specSummed),dim=-1,keepdim=True).values

        if self.mag:
            if gen: print('>>>>> Generating magnitude spectra')
            magnitude = torch.sqrt(specSummed[:,0,:]**2 + specSummed[:,1,:]**2).unsqueeze(1)
            mag_norm = magnitude / torch.max(torch.abs(magnitude),dim=-1,keepdim=True).values
            out = mag_norm
        else:
            out = spec_norm

        if gen==False:
            return out if not self.cropped else out[:,:,self.cropRange[0]:self.cropRange[1]]
        else:
            # if background_spectra:
            #     params = params[:,0:23]
            if self.cropped:
                spec_norm = spec_norm[:,:,self.cropRange[0]:self.cropRange[1]]
                out = out[:,:,self.cropRange[0]:self.cropRange[1]]
            return spec_norm, out, params


def fftshift(x, dim=None):
    assert(torch.is_tensor(x))
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [d // 2 for d in x.shape]
    else:
        shift = [x.shape[dim] // 2]
    return torch.roll(x, shift, dim)

def ifftshift(x, dim=None):
    assert(torch.is_tensor(x))
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [-(d // 2) for d in x.shape]
    else:
        shift = [-(x.shape[dim] // 2)]
    return torch.roll(x, shift, dim)

def complex_adjustment(input, theta):
    out = torch.empty_like(input, requires_grad=input.requires_grad)

    if theta.dim() <= 2:
        if input.dim()>=3.0:
            out[:,0,::] = input[:,0,::]*torch.cos(theta) - input[:,1,::]*torch.sin(theta)
            out[:,1,::] = input[:,0,::]*torch.sin(theta) + input[:,1,::]*torch.cos(theta)
        elif input.dim()==2:
            out[0,:] = input[0,:]*torch.cos(theta) - input[1,:]*torch.sin(theta)
            out[1,:] = input[0,:]*torch.sin(theta) + input[1,:]*torch.cos(theta)
    elif theta.dim() == 3:
        if input.dim()>=3.0:
            out[:,0,::] = input[:,0,::]*torch.cos(theta[:,0,:]) - input[:,1,::]*torch.sin(theta[:,0,:])
            out[:,1,::] = input[:,0,::]*torch.sin(theta[:,0,:]) + input[:,1,::]*torch.cos(theta[:,0,:])
        elif input.dim()==2:
            out[0,:] = input[0,:]*torch.cos(theta) - input[1,:]*torch.sin(theta)
            out[1,:] = input[0,:]*torch.sin(theta) + input[1,:]*torch.cos(theta)

    return out

