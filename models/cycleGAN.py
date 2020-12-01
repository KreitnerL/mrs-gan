from models.EntropyProfileLoss import EntropyProfileLoss
import torch
import torch.nn as nn
from collections import OrderedDict
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import define
T = torch.Tensor
from models.lr_scheduler import get_scheduler_G, get_scheduler_D


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

class CycleGANModel(BaseModel):
    """
    This class implements a CycleGAN model for learning domain to domain translation without paired data.
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def name(self):
        return 'CycleGAN'

    def __init__(self, opt, num_dimensions=2):
        super().__init__(opt)
        networks.set_num_dimensions(num_dimensions)

        nb = opt.batch_size
        size = opt.fineSize
        self.input_A: T = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B: T = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # Generators
        self.netG_A = define.define_modular_G(opt.input_nc, opt.output_nc, 
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, n_downsampling=opt.n_downsampling)
        self.netG_B = define.define_modular_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, n_downsampling=opt.n_downsampling)

        # Discriminators
        if self.isTrain:
            self.netD_A = define.define_D(opt, opt.input_nc,
                                            opt.ndf, opt.which_model_netD, 
                                            opt.n_layers_D, opt.norm, self.gpu_ids)
            self.netD_B = define.define_D(opt, opt.input_nc,
                                            opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids)

        if not self.opt.quiet:
            print('---------- Networks initialized -------------')
            define.print_network(self.netG_A)
            define.print_network(self.netG_B)
            if self.isTrain:
                define.print_network(self.netD_A)
                define.print_network(self.netD_B)
            print('-----------------------------------------------')

        # Load checkpoint
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_A, 'G_A', opt.epoch_count)
            self.load_network(self.netG_B, 'G_B', opt.epoch_count)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', opt.epoch_count)
                self.load_network(self.netD_B, 'D_B', opt.epoch_count)
            print('Loaded checkpoint', opt.epoch_count)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
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
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), 
                                        lr=opt.dlr, betas=(opt.beta2, 0.9))

        self.optimizers['Generator'] = self.optimizer_G
        self.optimizers['Discriminator'] = self.optimizer_D
        self.schedulers = [
            get_scheduler_G(self.optimizer_G, opt),
            get_scheduler_D(self.optimizer_D, opt)
        ]

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.\n
        input (dict): include the data itself and its metadata information.\n
        The option 'direction' can be used to swap domain A and domain B.
        """
        if 'A' in input:
            input_A: T = input['A']
            self.input_A.resize_(input_A.size()).copy_(input_A)

        if 'B' in input:
            input_B: T = input['B']
            self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = input['A_paths']

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        if self.opt.phase != 'val' or self.opt.AtoB:
            self.real_A = self.input_A
            self.fake_B = self.netG_A.forward(self.real_A)
            self.rec_A = self.netG_B.forward(self.fake_B)

        if self.opt.phase != 'val' or not self.opt.AtoB:
            self.real_B = self.input_B
            self.fake_A = self.netG_B.forward(self.real_B)
            self.rec_B = self.netG_A.forward(self.fake_A)

    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD: nn.Module, real: T, fake: T):
        """Calculate GAN loss for the discriminator\n
            netD (network)      -- the discriminator D\n
            real (tensor array) -- real images\n
            fake (tensor array) -- images generated by a generator\n
        Return the discriminator loss.\n
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        # The Discrimintator performs good when it return small numbers for real samples and big numbers for fake samples
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def calculate_G_loss(self):
        """Calculate the loss for generators G_A and G_B"""
        self.calculate_identity_loss()

        # GAN loss
        # The Generator performs good when the the discriminator return a small number for a fake, i.e. treats it like a real sample. => Aversarial to D loss
        # D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
        # Entropy loss
        if self.lambda_entropy != 0:
            self.loss_entropy_A: T = self.lambda_entropy * self.criterionEntropy.forward(self.rec_A, self.real_A)
            self.loss_entropy_B: T = self.lambda_entropy * self.criterionEntropy.forward(self.rec_B, self.real_B)
        else:
            self.loss_entropy_A = self.loss_entropy_B = 0


        # combined loss
        self.loss_G: T = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_entropy_A + self.loss_entropy_B
        return self.loss_G

    def calculate_identity_loss(self):
        """Calculates the idetity loss"""
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A: T = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B: T = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

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
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()

    def get_current_losses(self):
        D_A = self.loss_D_A.data
        G_A = self.loss_G_A.data
        Cyc_A = self.loss_cycle_A.data
        D_B = self.loss_D_B.data
        G_B = self.loss_G_B.data
        Cyc_B = self.loss_cycle_B.data
        G = self.loss_G
        if self.opt.lambda_identity > 0.0:
            idt_A = self.loss_idt_A.data
            idt_B = self.loss_idt_B.data
            return OrderedDict([('D_A', D_A), ('D_B', D_B), ('G', G), ('G_A', G_A), ('G_B', G_B), 
                                ('Cyc_B', Cyc_B), ('Cyc_A', Cyc_A), ('idt_A', idt_A), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('D_B', D_B), ('G', G),
                                ('G_A', G_A), ('G_B', G_B), ('Cyc_A', Cyc_A), ('Cyc_B', Cyc_B), ])

    def get_current_visuals(self):
        raise NotImplementedError()

    def save(self, label):
        """ Create a checkpoint of the current state of the model """
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def get_fake(self):
        if self.opt.AtoB:
            return self.fake_B
        else:
            return self.fake_A

