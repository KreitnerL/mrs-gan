import torch
import torch.nn as nn
import scipy.io as io
import numpy as np
from torch.nn import parameter
from models.auxiliaries.cubichermitesplines import CubicHermiteSplines
T = torch.Tensor

class PhysicsModel(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.roi = self.opt.roi
        # TODO make optional
        paths = [
            '/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/Para_for_spectra_gen.mat',
            '/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/conc_ranges_for_Linus.mat',
            '/home/kreitnerl/mrs-gan/spectra_generation/basis_spectra/units.mat'
        ]
        self.params = dict()
        for path in paths:
            parameters = io.loadmat(path)
            for key in parameters.keys():
                if str(key).startswith('_'):
                    continue
                self.params[key] = torch.FloatTensor(np.asarray(parameters[key], dtype=np.float64)).squeeze()
        
        self.register_buffer('max_per_met', torch.tensor([
            self.params['pch_max'],
            self.params['naa_max']
        ]).unsqueeze(0))

        self.register_buffer('cre_p', torch.ones(self.opt.batch_size,1))

        # Tensor of shape (1,M,2,L)
        basis_fids = torch.cat((
            self.params['fidCh'].unsqueeze(0),
            self.params['fidNaa'].unsqueeze(0),
            self.params['fidCr'].unsqueeze(0)
        ), dim=0).unsqueeze(0).cuda()
        self.register_buffer('basis_spectra', _export(basis_fids, roi=self.roi))

        self.opt.relativator = torch.max(torch.sqrt(self.basis_spectra[:,-2]**2 + self.basis_spectra[:,-1]**2))

        if self.opt.mag:
            index_real = [i%2==0 for i in range(self.basis_spectra.shape[1])]
            index_imag = [i%2==1 for i in range(self.basis_spectra.shape[1])]
            spectrum_real = self.basis_spectra[:,index_real,:]
            spectrum_imag = self.basis_spectra[:,index_imag,:]
            self.basis_spectra = torch.sqrt(spectrum_real**2 + spectrum_imag**2)

        self.register_buffer('magic_number', torch.tensor([
            3.0912,
            1.0221
        ]).unsqueeze(0))

    def forward(self, parameters: T):
        """
        Parameters:
        -----------
            - quantities (torch.Tensor): Tensor of shape (B,M) storing the metabolite parameters per sample
        Returns:
        --------
            - Tensor of shape (BxCxL) containing ideal spectrum
        """
        parameters = torch.cat([parameters*self.max_per_met, self.cre_p],1)
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
        return ideal_spectra/self.opt.relativator

    def get_num_out_channels(self):
        return 2

    def quantity_to_param(self, quantities: T):
        return quantities / self.max_per_met
    
    def param_to_quantity(self, params: T):
        return params * self.max_per_met.detach().cpu()

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
        - mag (bool): DEPRICATED! Use magnitude of basis spectra. Default=False
        - roi (slice): Final range the spectra should be cropped to. Default = no cropping
    
    Returns:
    --------
        - Tensor of shape (Bx2MxL) containing the basis spectra for each metabolite for each sample
    """
    # Recover Spectrum
    specSummed = fftshift(torch.fft(fids.transpose(3,2),2).transpose(3,2),-1)

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

# def angle(z: torch.Tensor, deg=False):
#     zreal = z[:,:,0,:]
#     zimag = z[:,:,1,:]

#     a = zimag.atan2(zreal)
#     if deg:
#         a = a * 180 / torch.from_numpy(np.pi)
#     return a

# def complex_adjustment(input, theta):
#     out = torch.empty_like(input, requires_grad=input.requires_grad)

#     if theta.dim() <= 2:
#         if input.dim()>=3.0:
#             out[:,0,::] = input[:,0,::]*torch.cos(theta) - input[:,1,::]*torch.sin(theta)
#             out[:,1,::] = input[:,0,::]*torch.sin(theta) + input[:,1,::]*torch.cos(theta)
#         elif input.dim()==2:
#             out[0,:] = input[0,:]*torch.cos(theta) - input[1,:]*torch.sin(theta)
#             out[1,:] = input[0,:]*torch.sin(theta) + input[1,:]*torch.cos(theta)
#     elif theta.dim() == 3:
#         if input.dim()>=3.0:
#             out[:,0,::] = input[:,0,::]*torch.cos(theta[:,0,:]) - input[:,1,::]*torch.sin(theta[:,0,:])
#             out[:,1,::] = input[:,0,::]*torch.sin(theta[:,0,:]) + input[:,1,::]*torch.cos(theta[:,0,:])
#         elif input.dim()==2:
#             out[0,:] = input[0,:]*torch.cos(theta) - input[1,:]*torch.sin(theta)
#             out[1,:] = input[0,:]*torch.sin(theta) + input[1,:]*torch.cos(theta)
#     elif theta.dim() == 4:
#         if input.dim()>=4.0:
#             out[:,:,0,::] = input[:,:,0,::]*torch.cos(theta[:,:,0,:]) - input[:,:,1,::]*torch.sin(theta[:,:,0,:])
#             out[:,:,1,::] = input[:,:,0,::]*torch.sin(theta[:,:,0,:]) + input[:,:,1,::]*torch.cos(theta[:,:,0,:])
#     return out

# def remove_zeroorderphase(spec: torch.Tensor):
#     a = angle (spec, deg=False)
#     return complex_adjustment(spec, -a)