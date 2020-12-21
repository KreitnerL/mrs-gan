from collections import OrderedDict
import itertools
from models.auxiliaries.EntropyProfileLoss import EntropyProfileLoss
from models.auxiliaries.physics_model import PhysicsModel

from models.cycleGAN_WGP import CycleGAN_WGP
from models.auxiliaries.lr_scheduler import get_scheduler_D, get_scheduler_G
from util.image_pool import ImagePool
import torch
import numpy as np
import util.util as util

from . import networks, define
T = torch.Tensor

class cycleGAN_WGP_REG(CycleGAN_WGP):
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

        
        self.netG_A = define.define_extractor(opt, opt.input_nc, opt.ndf, 3, opt.norm, self.gpu_ids, init_type=opt.init_type,
                                            cbam=opt.cbamD, output_nc=self.physicsModel.get_num_out_channels())
        self.netG_B = define.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                            opt.norm, not opt.no_dropout, self.gpu_ids, init_type=opt.init_type)
        
        if self.isTrain:
            self.netD_B = define.define_D(opt, opt.input_nc,
                                            opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)
        
        if not self.opt.quiet:
            print('---------- Networks initialized -------------')
            define.print_network(self.netG_A)
            define.print_network(self.netG_B)
            if self.isTrain:
                define.print_network(self.netD_B)
                self.save_network_architecture([self.netG_A, self.netG_B, self.netD_B])
            print('-----------------------------------------------')

        # Load checkpoint
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_A, 'G_A', opt.network_stage)
            self.load_network(self.netG_B, 'G_B', opt.network_stage)
            if self.isTrain:
                self.load_network(self.netD_B, 'D_B', opt.network_stage)
            print('Loaded checkpoint', opt.network_stage)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionIdt = torch.nn.MSELoss()
            self.criterionEntropy = EntropyProfileLoss(kernel_sizes=(2,3,4))
            # initialize optimizers
            if not opt.TTUR:
                self.opt.glr = opt.lr
                self.opt.dlr = opt.lr
                self.opt.n_epochs_gen_decay = opt.n_epochs_decay
                self.opt.n_epochs_dis_decay = opt.n_epochs_decay
            
            self.old_glr = opt.lr
            self.old_dlr = opt.lr
            
            self.init_optimizers(opt)

            # Set loss weights
            self.lambda_idt = self.opt.lambda_identity
            self.lambda_A = self.opt.lambda_A
            self.lambda_B = self.opt.lambda_B
            self.lambda_entropy = self.opt.lambda_entropy

    def init_optimizers(self, opt):
        """
        Initialize optimizers and learning rate schedulers
        """
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.glr, betas=(opt.beta1, 0.9))
        self.optimizer_D = torch.optim.Adam(self.netD_B.parameters(), lr=opt.dlr, betas=(opt.beta2, 0.9))

        self.optimizers['Generator'] = self.optimizer_G
        self.optimizers['Discriminator'] = self.optimizer_D
        self.schedulers = [
            get_scheduler_G(self.optimizer_G, opt),
            get_scheduler_D(self.optimizer_D, opt)
        ]

    def set_fix_inputs(self, basis_spectra: T):
        self.basis_spectra = basis_spectra.cuda()

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
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
        # Entropy loss
        if self.lambda_entropy != 0:
            self.loss_entropy_A: T = self.lambda_entropy * self.criterionEntropy.forward(self.rec_A, self.real_A)
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

    def save(self, label):
        """ Create a checkpoint of the current state of the model """
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def get_current_losses(self):
        D_B = self.loss_D_B.data
        G_B = self.loss_G_B.data
        Cyc_A = self.loss_cycle_A.data
        Cyc_B = self.loss_cycle_B.data
        entropy_A = self.loss_entropy_A
        G = self.loss_G
        return OrderedDict([('G', G), ('D_B', D_B), ('Cyc_B', Cyc_B), ('Cyc_A', Cyc_A), ('G_B', G_B), ('Entropy_A', entropy_A) ])

    def get_current_visuals(self):
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None
        if hasattr(self, 'real_A'):
            x = np.linspace(*self.opt.ppm_range, self.opt.full_data_length)[self.opt.roi]
            real_A = util.get_img_from_fig(x, self.real_A[0:1].data, 'PPM')
            fake_B = util.get_img_from_fig(x, self.physicsModel.forward(self.fake_B)[0:1].data, 'PPM')
            rec_A = util.get_img_from_fig(x, self.rec_A[0:1].data, 'PPM')
        if hasattr(self, 'real_B'):
            x = np.linspace(*self.opt.ppm_range, self.opt.full_data_length)[self.opt.roi]
            real_B = util.get_img_from_fig(x, self.physicsModel.forward(self.real_B)[0:1].data, 'PPM')
            fake_A = util.get_img_from_fig(x, self.fake_A[0:1].data, 'PPM')
            rec_B = util.get_img_from_fig(x, self.physicsModel.forward(self.rec_B)[0:1].data, 'PPM')

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def get_fake(self):
        return self.physicsModel.param_to_quantity(self.fake_B.detach().cpu())