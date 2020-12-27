from collections import OrderedDict
import itertools

from models.auxiliaries.auxiliary import relativeMELoss
from models.auxiliaries.EntropyProfileLoss import EntropyProfileLoss
from models.auxiliaries.physics_model import PhysicsModel

from models.w_cycleGAN import W_CycleGAN
from models.auxiliaries.lr_scheduler import get_scheduler_D, get_scheduler_G
from util.image_pool import ImagePool
import torch
import numpy as np
import util.util as util

from . import networks, define
T = torch.Tensor

class cycleGAN_WGP_REG(W_CycleGAN):
    """
    This class implements the novel cyleGAN for unsupervised spectral quantification tasks
    """

    def __init__(self, opt, physicsModel):
        opt.lambda_identity = 0
        self.physicsModel: PhysicsModel = physicsModel
        self.physicsModel.cuda()
        super().__init__(opt)
    
    def name(self):
        return 'CycleGAN_REG'

    def init(self, opt):
        nb = opt.batch_size
        self.input_A: T = self.Tensor(nb, opt.input_nc, opt.data_length)
        self.input_B: T = self.Tensor(nb, self.physicsModel.get_num_out_channels(), 1)

        
        self.netG_A = define.define_extractor(opt.input_nc, self.physicsModel.get_num_out_channels(), opt.data_length, opt.nef, opt.n_layers_E,
                                            opt.norm, self.gpu_ids, cbam=True)
        self.netG_B = define.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                            opt.norm, not opt.no_dropout, self.gpu_ids, init_type=opt.init_type)
        
        if opt.isTrain:
            self.netD_B = define.define_D(opt, opt.input_nc,
                                            opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)
        self.networks = [self.netG_A, self.netD_B, self.netD_B]

        if opt.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionEntropy = EntropyProfileLoss(kernel_sizes=(2,3,4,5))

    def init_optimizers(self, opt):
        """
        Initialize optimizers and learning rate schedulers
        """
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.glr, betas=(0, 0.9))
        self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(), lr=opt.dlr, betas=(0, 0.9))

        self.optimizers['Generator'] = self.optimizer_G
        self.optimizers['Discriminator'] = self.optimizer_D
        self.schedulers = [
            get_scheduler_G(self.optimizer_G, opt),
            get_scheduler_D(self.optimizer_D, opt)
        ]

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        self.real_A = self.input_A
        self.fake_B = self.netG_A.forward(self.real_A)
        ideal_spectra = self.physicsModel.forward(self.fake_B)
        self.rec_A = self.netG_B.forward(ideal_spectra)

        if self.opt.phase != 'val':
            self.real_B = self.physicsModel.quantity_to_param(self.input_B)
            ideal_spectra = self.physicsModel.forward(self.real_B)
            self.fake_A = self.netG_B.forward(ideal_spectra)
            self.rec_B = self.netG_A.forward(self.fake_A)

    def calculate_G_loss(self):
        """Calculate the loss for generators G_A and G_B"""
        # GAN loss
        # The Generator performs good when the the discriminator return a small number for a fake, i.e. treats it like a real sample. => Aversarial to D loss
        # D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        # Entropy loss
        if self.opt.lambda_entropy != 0:
            entropy_loss, content_loss = self.criterionEntropy.forward(self.rec_A, self.real_A)
            self.loss_entropy_A: T = self.opt.lambda_entropy * (entropy_loss + content_loss)
        else:
            self.loss_entropy_A = 0

        # combined loss
        self.loss_G: T = self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_entropy_A
        return self.loss_G

    def optimize_parameters(self, optimize_G=True, optimize_D=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward() # compute fake images and reconstruction images.
        # G_A and G_B
        if optimize_G:
            self.optimizer_G.zero_grad()
            self.calculate_G_loss().backward()
            self.optimizer_G.step()
        if optimize_D:
            # D_A and D_B
            self.optimizer_D.zero_grad()
            self.backward_D_B()
            self.optimizer_D.step()

    def get_current_losses(self):
        D_B = self.loss_D_B.detach()
        G_B = self.loss_G_B.detach()
        Cyc_A = self.loss_cycle_A.detach()
        Cyc_B = self.loss_cycle_B.detach()
        entropy_A = self.loss_entropy_A.detach()
        G = self.loss_G.detach()
        return OrderedDict([('G', G), ('D_B', D_B), ('Cyc_B', Cyc_B), ('Cyc_A', Cyc_A), ('G_B', G_B), ('Entropy_A', entropy_A) ])

    def get_current_visuals(self):
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None

        if self.label_A.numel():
            self.real_B = self.label_A
            self.forward()

        x = np.linspace(*self.opt.ppm_range, self.opt.full_data_length)[self.opt.roi]
        real_A = util.get_img_from_fig(x, self.real_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)
        fake_B = util.get_img_from_fig(x, self.physicsModel.forward(self.fake_B)[0:1].detach(), 'PPM', magnitude=self.opt.mag)
        rec_A = util.get_img_from_fig(x, self.rec_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)

        if hasattr(self, 'real_B'):
            real_B = util.get_img_from_fig(x, self.physicsModel.forward(self.real_B)[0:1].detach(), 'PPM', magnitude=self.opt.mag)
            fake_A = util.get_img_from_fig(x, self.fake_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)
            rec_B = util.get_img_from_fig(x, self.physicsModel.forward(self.rec_B)[0:1].detach(), 'PPM', magnitude=self.opt.mag)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def get_fake(self):
        return self.physicsModel.param_to_quantity(self.fake_B.detach().cpu())