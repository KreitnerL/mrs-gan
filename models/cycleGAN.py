from util.image_pool import ImagePool
from validation_networks.MLP.MLP import MLP
from models.auxiliaries.physics_model_interface import PhysicsModel
from models.auxiliaries.FeatureProfileLoss import FeatureProfileLoss
import torch
import torch.nn as nn
from collections import OrderedDict
import itertools
from . import networks, define
import models.auxiliaries.auxiliary as auxiliary
import numpy as np
from collections import OrderedDict
import util.util as util
from models.auxiliaries.lr_scheduler import get_scheduler_G, get_scheduler_D
import os

T = torch.Tensor

class CycleGAN():
    """
    This class implements a CycleGAN model for learning domain to domain translation without paired data.
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def name(self):
        return 'CycleGAN'

    def __init__(self, opt, physicsModel: PhysicsModel, num_dimensions=1):
        auxiliary.set_num_dimensions(num_dimensions)
        self.physicsModel = physicsModel
        self.physicsModel.cuda()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizers = dict()
        self.schedulers = []
        self.init(opt)
        if opt.isTrain:
            self.old_glr = opt.lr
            self.old_dlr = opt.lr
            self.init_optimizers(opt)
            self.save_network_architecture(self.networks)
        self.init_loss_array()
        if not self.opt.quiet:
            print('---------- Networks initialized -------------')
            for network in self.networks:
                define.print_network(network)
            print('-----------------------------------------------')

    def init(self, opt):
        nb = opt.batch_size
        size = opt.fineSize
        self.input_A: T = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B: T = self.Tensor(nb, opt.output_nc, size, size)

        self.val_network = MLP(self.opt.val_path, gpu=self.opt.gpu_ids[0], in_out= (opt.input_nc * opt.data_length, opt.output_nc), batch_size=opt.batch_size)
        assert self.val_network.pretrained

        # Generators
        self.netG_A = define.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamG)
        self.netG_B = define.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamG)

        self.networks = [self.netG_A, self.netG_B]
        # Discriminators
        if opt.isTrain:
            self.netD_A = define.define_D(opt, opt.input_nc, opt.ndf, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)
            self.netD_B = define.define_D(opt, opt.input_nc, opt.ndf, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)
            self.networks.extend([self.netD_A, self.netD_B])

        if opt.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionEntropy = FeatureProfileLoss(kernel_sizes=(2,3,4,5))

    def init_loss_array(self):
        self.loss_names = [
            'loss_G',
            'loss_D_A',
            'loss_D_B',
            'loss_G_A',
            'loss_G_B',
            'loss_cycle_A',
            'loss_cycle_B',
            'loss_feat_A',
            'loss_feat_B',
            'loss_idt_A',
            'loss_idt_B'
        ]
        for loss in self.loss_names:
            setattr(self, loss, 0)

    def init_optimizers(self, opt):
        """
        Initialize optimizers and learning rate schedulers
        """
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.glr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), 
                                        lr=opt.dlr, betas=(opt.beta1, opt.beta2))

        self.optimizers['Generator'] = self.optimizer_G
        self.optimizers['Discriminator'] = self.optimizer_D
        self.schedulers = [
            get_scheduler_G(self.optimizer_G, opt),
            get_scheduler_D(self.optimizer_D, opt)
        ]

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = {}
        for name, optimizer in self.optimizers.items():
            old_lr[name] = optimizer.param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        for name, optimizer in self.optimizers.items():
            print(name, ': learning rate %.7f -> %.7f' % (old_lr[name], optimizer.param_groups[0]['lr']))


    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.\n
        input (dict): include the data itself and its metadata information.\n
        The option 'direction' can be used to swap domain A and domain B.
        """
        if 'A' in input:
            input_A: T = input['A']
            self.label_A: T = input['label_A']
            self.input_A.resize_(input_A.size()).copy_(input_A)

        if 'B' in input:
            input_B: T = input['B']
            self.input_B.resize_(input_B.size()).copy_(input_B)

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        self.real_A = self.input_A
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        if self.opt.phase != 'val':
            self.real_B = self.physicsModel.forward(self.physicsModel.quantity_to_param(self.input_B))
            self.fake_A = self.netG_B.forward(self.real_B)
            self.rec_B = self.netG_A.forward(self.fake_A)

    def test(self):
        with torch.no_grad():
            if self.label_A.numel():
                self.input_B.resize_(self.label_A.size()).copy_(self.label_A)
            else:
                print('WARNING: Paired forward not possile: No label_A found.')
            self.forward()

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
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        # Feature loss
        if self.opt.lambda_feat != 0:
            entropy_loss, content_loss = self.criterionEntropy.forward(self.rec_A, self.real_A)
            self.loss_feat_A: T = self.opt.lambda_feat * (entropy_loss + content_loss)
            entropy_loss, content_loss = self.criterionEntropy.forward(self.rec_B, self.real_B)
            self.loss_feat_B: T = self.opt.lambda_feat * (entropy_loss + content_loss)
        else:
            self.lambda_feat_A = self.lambda_feat_B = 0


        # combined loss
        self.loss_G: T = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_feat_A + self.loss_feat_B
        return self.loss_G

    def calculate_identity_loss(self):
        """Calculates the idetity loss"""
        if self.opt.lambda_identity > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A: T = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B: T = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity
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
        d = OrderedDict()
        for loss in self.loss_names:
            val: T = getattr(self, loss)
            if val != 0:
                d[loss] = val.detach()
        return d

    def paired_forward(self):
        if self.label_A.numel():
            self.input_B.resize_(self.label_A.size()).copy_(self.label_A)
        else:
            print('WARNING: Paired forward not possile: No label_A found.')
        self.forward()

    def get_current_visuals(self, get_all = False):
        self.test()
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None

        x = np.linspace(*self.opt.ppm_range, self.opt.full_data_length)[self.opt.roi]
        real_A = util.get_img_from_fig(x, self.real_A[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')
        fake_B = util.get_img_from_fig(x, self.fake_B[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')
        rec_A = util.get_img_from_fig(x, self.rec_A[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')
        if hasattr(self, 'real_B'):
            real_B = util.get_img_from_fig(x, self.real_B[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')
            fake_A = util.get_img_from_fig(x, self.fake_A[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')
            rec_B = util.get_img_from_fig(x, self.rec_B[0:1].detach(), 'PPM', magnitude=self.opt.representation == 'mag')

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                        ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def get_items(self):
        items = dict()
        items['real_A_spectra'] = self.real_A.detach().cpu().numpy()
        items['fake_B_spectra'] = self.fake_B.detach().cpu().numpy()
        items['rec_A_spectra'] = self.rec_A.detach().cpu().numpy()

        items['real_B_spectra'] = self.real_B.detach().cpu().numpy()
        items['fake_A_spectra'] = self.fake_A.detach().cpu().numpy()
        items['rec_B_spectra'] = self.rec_B.detach().cpu().numpy()

        return items

    def create_checkpoint(self, path, d=None):
        states = list(map(lambda x: x.cpu().state_dict(), self.networks))
        checkpoint = {
            "networks": states
        }
        if d is not None:
            checkpoint.update(d)
        torch.save(checkpoint, path)
        for x in self.networks:
            x.cuda()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        states = checkpoint.pop('networks')
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(states[i])
        print('Loaded checkpoint successfully')
        return checkpoint

    def save_network_architecture(self, networks):
        save_filename = 'architecture.txt'
        save_path = os.path.join(self.save_dir, save_filename)

        architecture = ''
        for n in networks:
            architecture += str(n) + '\n'
        with open(save_path, 'w') as f:
            f.write(architecture)
            f.flush()
            f.close()

    def get_prediction(self) -> np.ndarray:
        return self.val_network.predict(self.get_predicted_spectra())

    def get_predicted_spectra(self) -> np.ndarray:
        return self.fake_B.detach().cpu().numpy()