from models.auxiliaries.physics_model_interface import PhysicsModel
import torch
import torch.nn as nn
# import torch.fft
import scipy.io as io
import numpy as np
import os
from models.auxiliaries.cubichermitesplines import CubicHermiteSplines
T = torch.Tensor

class MRSPhysicsModel(PhysicsModel):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.roi = self.opt.roi
        self.standard_β = -0.000416455078125
        opt.mrs_physics_model = self

        self.params = dict()
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'spectra_generation_params.mat')
        parameters = io.loadmat(filename)
        for key in parameters.keys():
            if str(key).startswith('_'):
                continue
            self.params[key] = torch.FloatTensor(np.asarray(parameters[key], dtype=np.float64)).squeeze()
    
        self.register_buffer('max_per_met', torch.tensor([
            self.params['pch_max'],
            self.params['naa_max']
        ]).unsqueeze(0))
        self.register_buffer('min_per_met', torch.tensor([
            self.params['pch_min'],
            self.params['naa_min']
        ]).unsqueeze(0))

        # self.register_buffer('cre_p', torch.ones(1,1, dtype=torch.float64))
        self.register_buffer('cre_p', torch.ones(1,1))

        # Tensor of shape (1,M,2,L)
        self.basis_fids = torch.cat((
            self.params['fidCh'].unsqueeze(0) / 3.0912,
            self.params['fidNaa'].unsqueeze(0) / 1.0221,
            self.params['fidCr'].unsqueeze(0)
        ), dim=0).unsqueeze(0).cuda()
        self.register_buffer('basis_spectra', _export(self.basis_fids, roi=self.roi))

        if self.opt.mag:
            index_real = [i%2==0 for i in range(self.basis_spectra.shape[1])]
            index_imag = [i%2==1 for i in range(self.basis_spectra.shape[1])]
            spectrum_real = self.basis_spectra[:,index_real,:]
            spectrum_imag = self.basis_spectra[:,index_imag,:]
            self.basis_spectra = torch.sqrt(spectrum_real**2 + spectrum_imag**2)

    def build_spectra(self, quantities: T, β_min = 1.0, β_max = 1.0, snr_min = 1, snr_max = 1):
        """
        Generates the simulated spectra with the given quantities using the given parameters. Formula: \n
        s[l] = FFT(Σ_{m∈M} (λ_m * b_m[t] * exp[β_m*l] + n*g[l]

        where n*g[l] is chosen to that the required SNR is achieved.

        Params:
        ------
            - quantities: Tensor of size (NxM) - The concentration of each metabolite for each sample
            - β_min: float - the minimum line broadening factor. Default = 1.0
            - β_max: float - the maximum line broadening factor. Default = 1.0
            - snr_min: float- the minimal SNR in dB. Default = 1.0
            - snr_max: float- the maximal SNR in dB. Default = 1.0

        Returns:
        -------
            A Tensor of shape Nx2xL, containing N spectra with real and imaginary channel and L datapoints (cf. self.roi).
        """
        fids: T = self.basis_fids
        M = fids.shape[1]
        N = quantities.shape[0]
        L = fids.shape[-1]

        # Line Broadening
        x = torch.arange(L, device='cuda').unsqueeze(0).repeat(M, 1)
        β_factor = (β_max-β_min) * torch.rand(N, M, device='cuda') + β_min
        β = self.standard_β / β_factor
        x = β.unsqueeze(-1) * x.unsqueeze(0)
        line_broadening = torch.exp(x) # Invidual line broadening function per metabolite per sample
        fids = fids * line_broadening.unsqueeze(2) # Nx3x2x2048

        # Concentration scaling
        fids = fids * quantities.unsqueeze(-1).unsqueeze(-1)
        fid_sum = fids.sum(1) # Nx2x2048

        # FFT
        spectra = fftshift(torch.fft(fid_sum.transpose(2,1),1).transpose(2,1),-1)
        spectra = _resample_(spectra, 1024).cuda() # Nx2xROI

        # Noise SNR_db = 10*log10(P_s/ P_n)
        snr = (snr_max-snr_min) * torch.rand(N, device='cuda') + snr_min
        P_signal = (spectra**2).mean(-1).mean(-1) # Signal power
        P_noise = P_signal / (10**(snr/10)) # Noise power
        noise = torch.distributions.normal.Normal(0, torch.sqrt(P_noise)).sample((2, spectra.shape[-1])).permute(2,0,1) # Nx2xROI
        noisy_spec = spectra + noise

        # Normalize
        norm_spectra = self.normalize(noisy_spec[:,:,self.roi])

        return norm_spectra


    def forward(self, parameters: T):
        """
        Parameters:
        -----------
            - quantities (torch.Tensor): Tensor of shape (B,M) storing the metabolite parameters per sample
        Returns:
        --------
            - Tensor of shape (BxCxL) containing ideal spectrum
        """
        parameters = torch.cat([parameters*self.max_per_met+self.min_per_met, self.cre_p.repeat(parameters.shape[0],1)],1)
        if self.opt.mag:
            modulated_basis_spectra = parameters.unsqueeze(-1)*self.basis_spectra
            ideal_spectra = modulated_basis_spectra.sum(1, keepdim=True)
        else:
            modulated_basis_spectra = torch.repeat_interleave(parameters, 2, 1).unsqueeze(-1) * self.basis_spectra
            index_real = [i%2==0 for i in range(modulated_basis_spectra.shape[1])]
            index_imag = [i%2==1 for i in range(modulated_basis_spectra.shape[1])]
            ideal_spectra = torch.cat([
                modulated_basis_spectra[:,index_real,:].sum(1, keepdim=True), 
                modulated_basis_spectra[:,index_imag,:].sum(1, keepdim=True)
            ], dim=1)
        return self.normalize(ideal_spectra)

    def normalize(self, x: T):
        shape = x.shape
        x = x.reshape(shape[0],-1)
        x = x/abs(x).max(1, keepdim=True)[0]
        return x.view(*shape)

    def get_num_out_channels(self):
        return 2

    def quantity_to_param(self, quantities: T):
        return (quantities-self.min_per_met) / self.max_per_met
    
    def param_to_quantity(self, params: T):
        return params * self.max_per_met.detach().cpu() + self.min_per_met.detach().cpu()

    def plot_basisspectra(self, path, plot_sum=False):
        import matplotlib.pyplot as plt
        x = np.linspace(self.opt.ppm_range[0], self.opt.ppm_range[-1], self.opt.full_data_length)[self.opt.roi]
        plt.figure()
        if self.opt.mag:
            s = self.basis_spectra[0].detach().cpu().numpy()
            s = s/np.amax(s)
            if plot_sum:
                basis_spectra_sum = np.sum(s, axis=0)
                plt.plot(x, basis_spectra_sum, color='gray')
            plt.plot(x, s.transpose())
            if plot_sum:
                labels = ['sum','cho', 'naa', 'cre']
            else:
                labels = ['cho', 'naa', 'cre']
        else:
            s = self.basis_spectra[0].detach().cpu().numpy()
            s_sum = torch.tensor([s[0]+s[2]+s[4], s[1]+s[3]+s[5]])
            plt.plot(x, s_sum.transpose(0,1), color='gray')
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            for i in range(int(len(s)/2)):
                plt.plot(x, s[i*2:i*2+2].transpose(), color=colors[i])
            if plot_sum:
                labels = ['sum', '_nolegend_','cho', '_nolegend_', 'naa', '_nolegend_', 'cre', '_nolegend_']
            else:
                labels = ['cho', '_nolegend_', 'naa', '_nolegend_', 'cre', '_nolegend_']

        plt.legend(labels)
        plt.xlim(x[0], x[-1])
        plt.xlabel('ppm')
        plt.title('%sBasisspectra'%('Magnitude ' if self.opt.mag else ''))
        plt.savefig(path, format='png', bbox_inches='tight')

#################################################################
#   HELPER FUNCTIONS                                            #
#################################################################

def _resample_(signal, length=1024, crop_start=865.6, crop_end=1357.12):
    # print('cropRange: ',self.cropRange)
    new = torch.linspace(start=crop_start, end=crop_end, steps=int(length)).to(signal.device)
    xaxis = torch.arange(signal.shape[-1])
    chs_interp = CubicHermiteSplines (xaxis, signal)
    return chs_interp.interp(new)

def _export(fids: T, roi=slice(None,None)):
    """
    Performs crop(resample(crop(corm(fftshift(fft(fids))))))
    Parameters:
    ----------
        - fids (torch.Tensor): Tensor of shape (B,M,C,L) containing the modulated basis FIDs
        - roi (slice): Final range the spectra should be cropped to. Default = no cropping
    
    Returns:
    --------
        - Tensor of shape (Bx2MxL) containing the basis spectra for each metabolite for each sample
    """
    # Recover Spectrum
    specSummed = fftshift(torch.fft(fids.transpose(3,2),1).transpose(3,2),-1)

    # Normalize Spectra by dividing by the norm of the area under the 3 major peaks
    # channel_max = torch.max(torch.abs(specSummed),dim=-1, keepdim=True).values
    # sample_max:T = torch.max(torch.abs(channel_max),dim=-2, keepdim=True).values
    # spec_norm = specSummed / sample_max#.repeat(int(self.l), dim=2)
    spec_norm = specSummed
    # spec_norm = specSummed / torch.max(torch.abs(specSummed),dim=-1,keepdim=True).values
    out = spec_norm.reshape(spec_norm.shape[0], 2*spec_norm.shape[1], spec_norm.shape[3])
    
    return _resample_(out, 1024).cuda()[:,:,roi]

def fftshift(x, dim=None):
    assert(torch.is_tensor(x))
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [d // 2 for d in x.shape]
    else:
        shift = [x.shape[dim] // 2]
    return torch.roll(x, shift, dim) 