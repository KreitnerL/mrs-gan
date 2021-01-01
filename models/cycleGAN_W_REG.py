from collections import OrderedDict
import itertools

from models.auxiliaries.auxiliary import relativeMELoss
from models.auxiliaries.FeatureProfileLoss import FeatureProfileLoss
from models.auxiliaries.physics_model import PhysicsModel

from models.cycleGAN_W import CycleGAN_W
from models.auxiliaries.lr_scheduler import get_scheduler_D, get_scheduler_G
from util.image_pool import ImagePool
import torch
import numpy as np
import util.util as util

from . import networks, define
T = torch.Tensor

class cycleGAN_W_REG(CycleGAN_W):
    """
    This class implements the novel cyleGAN for unsupervised spectral quantification tasks
    """

    def __init__(self, opt, physicsModel: PhysicsModel):
        opt.lambda_identity = 0
        super().__init__(opt, physicsModel)
    
    def name(self):
        return 'CycleGAN_REG'

    def init(self, opt):
        nb = opt.batch_size
        self.input_A: T = self.Tensor(nb, opt.input_nc, opt.data_length)
        self.input_B: T = self.Tensor(nb, self.physicsModel.get_num_out_channels(), 1)

        
        self.netG_A = define.define_extractor(opt.input_nc, self.physicsModel.get_num_out_channels(), opt.data_length, opt.nef, opt.n_layers_E,
                                            opt.norm, self.gpu_ids, cbam=True)
        self.netG_B = define.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                            opt.norm, self.gpu_ids, init_type=opt.init_type)
        # self.netG_B = define.define_G_MLP(opt.input_nc, opt.data_length, opt.nef, opt.n_layers_E, self.gpu_ids)
        self.networks = [self.netG_A, self.netG_B]
        
        if opt.isTrain:
            self.netD_B = define.define_D(opt, opt.input_nc, opt.ndf, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionEntropy = FeatureProfileLoss(kernel_sizes=(2,3,4,5))
            self.networks.extend([self.netD_B])

    def init_optimizers(self, opt):
        """
        Initialize optimizers and learning rate schedulers
        """
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.glr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(), lr=opt.dlr, betas=(opt.beta1, opt.beta2))

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
        # Feature loss
        if self.opt.lambda_feat != 0:
            entropy_loss, content_loss = self.criterionEntropy.forward(self.rec_A, self.real_A)
            self.loss_feat_A: T = self.opt.lambda_feat * (entropy_loss + content_loss)
        else:
            self.loss_feat_A = 0

        # combined loss
        self.loss_G: T = self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_feat_A
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

    def get_current_visuals(self):
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None

        if self.label_A.numel():
            self.input_B.resize_(self.label_A.size()).copy_(self.label_A)
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

    def get_prediction(self) -> np.ndarray:
        return self.physicsModel.param_to_quantity(self.fake_B.detach().cpu()).numpy()

    def get_predicted_spectra(self) -> np.ndarray:
        return self.physicsModel.forward(self.fake_B).detach().cpu().numpy()