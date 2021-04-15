import itertools
from collections import OrderedDict
import torch
import numpy as np
import util.util as util
from models.auxiliaries.lr_scheduler import get_scheduler_D, get_scheduler_G
from models import networks
from util.image_pool import ImagePool
from models.define import define_D, define_splitter, define_styleGenerator
from models.auxiliaries.physics_model import PhysicsModel
from models.auxiliaries.FeatureProfileLoss import FeatureProfileLoss
from models.cycleGAN_W import CycleGAN_W
T = torch.Tensor

class CycleGAN_REG(CycleGAN_W):
    """
    This class implements the novel CycleGAN architecture for arbitrary unsupervised regression task
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

        style_nc = 16
        self.splitter = define_splitter(opt.input_nc, opt.data_length, self.physicsModel.get_num_out_channels(), opt.ngf, 16, norm=opt.norm, gpu_ids=self.gpu_ids)
        self.styleGenerator = define_styleGenerator(opt.input_nc, opt.input_nc, 64, gpu_ids=self.gpu_ids)
        self.networks = [self.splitter, self.styleGenerator]

        if opt.isTrain:
            self.netD_B = define_D(opt, opt.input_nc, opt.ndf, opt.n_layers_D, 
                                            opt.norm, self.gpu_ids, init_type=opt.init_type, cbam=opt.cbamD)
            self.fake_A_pool = ImagePool(opt.pool_size)
            # TODO create option
            self.style_cache = ImagePool(opt.batch_size)

            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode, tensor=self.Tensor)
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionEntropy = FeatureProfileLoss(kernel_sizes=(2,3,4,5))
            self.networks.extend([self.netD_B])

    def init_optimizers(self, opt):
        """
        Initialize optimizers and learning rate schedulers
        """
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.splitter.parameters(), self.styleGenerator.parameters()),
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
        self.fake_params, self.fake_style = self.splitter.forward(self.real_A)
        ideal_spectra = self.physicsModel.forward(self.fake_params)
        self.rec_A = self.styleGenerator.forward(ideal_spectra, self.fake_style)

        if self.opt.phase != 'val':
            self.real_params = self.physicsModel.quantity_to_param(self.input_B)
            ideal_spectra = self.physicsModel.forward(self.real_params)

            self.real_style: "T" = self.style_cache.query(self.fake_style.detach())
    
            self.fake_A = self.styleGenerator.forward(ideal_spectra, self.real_style)
            self.rec_params, self.rec_style = self.splitter.forward(self.fake_A)

    def calculate_G_loss(self):
        """Calculate the loss for the splitter and the styleGenerator"""
        # GAN loss
        # The Generator performs good when the the discriminator return a small number for a fake, i.e. treats it like a real sample. => Aversarial to D loss
        # D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A: T = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        # Backward cycle loss
        self.loss_cycle_B: T = (self.criterionCycle(self.fake_params, self.fake_params) + self.criterionCycle(self.real_style, self.real_style)) * self.opt.lambda_B
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

    def get_current_visuals(self, get_all = False):
        self.test()
        real_A = real_B = fake_A = fake_B = rec_A = rec_B = x = None

        x = np.linspace(*self.opt.ppm_range, self.opt.full_data_length)[self.opt.roi]
        real_A = util.get_img_from_fig(x, self.real_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)
        fake_B = util.get_img_from_fig(x, self.physicsModel.forward(self.fake_params)[0:1].detach(), 'PPM', magnitude=self.opt.mag)
        rec_A = util.get_img_from_fig(x, self.rec_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)

        if hasattr(self, 'real_params'):
            real_B = util.get_img_from_fig(x, self.physicsModel.forward(self.real_params)[0:1].detach(), 'PPM', magnitude=self.opt.mag)
            fake_A = util.get_img_from_fig(x, self.fake_A[0:1].detach(), 'PPM', magnitude=self.opt.mag)
            rec_B = util.get_img_from_fig(x, self.physicsModel.forward(self.rec_params)[0:1].detach(), 'PPM', magnitude=self.opt.mag)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def get_items(self):
        items = dict()
        items['real_A_spectra'] = self.real_A.detach().cpu().numpy()
        items['fake_B_quantity'] = self.physicsModel.param_to_quantity(self.fake_params.detach().cpu()).numpy()
        items['fake_B_spectra'] = self.physicsModel.forward(self.fake_params).detach().cpu().numpy()
        items['rec_A_spectra'] = self.rec_A.detach().cpu().numpy()

        items['real_B_quantity'] = self.physicsModel.param_to_quantity(self.real_params.detach().cpu()).numpy()
        items['real_B_spectra'] = self.physicsModel.forward(self.real_params).detach().cpu().numpy()
        items['fake_A_spectra'] = self.fake_A.detach().cpu().numpy()
        items['rec_B_quantity'] =  self.physicsModel.param_to_quantity(self.rec_params.detach().cpu()).numpy()
        items['rec_B_spectra'] = self.physicsModel.forward(self.rec_params).detach().cpu().numpy()

        return items

    def get_prediction(self) -> np.ndarray:
        return self.physicsModel.param_to_quantity(self.fake_params.detach().cpu()).numpy()

    def get_predicted_spectra(self) -> np.ndarray:
        return self.physicsModel.forward(self.fake_params).detach().cpu().numpy()
