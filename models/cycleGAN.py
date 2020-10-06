import torch
import torch.nn as nn
from collections import OrderedDict
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
T = torch.Tensor


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

class CycleGANModel(BaseModel):
    """
    This class implements a CycleGAN model for learning 1d signal translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def name(self):
        return 'CycleGAN'

    def __init__(self, opt):
        super().__init__(opt)
        networks.set_num_dimensions(1)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A: T = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B: T = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # Generators
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, 
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)

        # Discriminators
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc,
                                            opt.ndf, opt.which_model_netD, 
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc,
                                            opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                                            opt.norm, use_sigmoid, self.gpu_ids)

        # Load checkpoint
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # Set loss weights
        self.lambda_idt = self.opt.identity
        self.lambda_A = self.opt.lambda_A
        self.lambda_B = self.opt.lambda_B

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.\n
        input (dict): include the data itself and its metadata information.\n
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        input_A: T = input['A' if AtoB else 'B']
        input_B: T = input['B' if AtoB else 'A']
        # TODO check if clone needed
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        self.real_A = self.input_A
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = self.input_B
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        pass

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

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
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

        # GAN loss
        # D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B

        # combined loss
        self.loss_G: T = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward() # compute fake images and reconstruction images.
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data
        G_A = self.loss_G_A.data
        Cyc_A = self.loss_cycle_A.data
        D_B = self.loss_D_B.data
        G_B = self.loss_G_B.data
        Cyc_B = self.loss_cycle_B.data
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data
            idt_B = self.loss_idt_B.data
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        x = list(range(self.real_A.size()[-1]))
        real_A = util.get_img_from_fig(x, self.real_A.data, 'PPM')
        fake_B = util.get_img_from_fig(x, self.fake_B.data, 'PPM')
        rec_A = util.get_img_from_fig(x, self.rec_A.data, 'PPM')
        real_B = util.get_img_from_fig(x, self.real_B.data, 'PPM')
        fake_A = util.get_img_from_fig(x, self.fake_A.data, 'PPM')
        rec_B = util.get_img_from_fig(x, self.rec_B.data, 'PPM')

        if self.opt.identity > 0.0:
            idt_A = util.get_img_from_fig(x, self.idt_A.data, 'PPM')
            idt_B = util.get_img_from_fig(x, self.idt_B.data, 'PPM')
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def save(self, label):
        """ Create a checkpoint of the current state of the model """
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

