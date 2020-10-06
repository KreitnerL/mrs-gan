from models.cycleGAN import CycleGANModel
import torch
from collections import OrderedDict
import util.util as util
from . import networks


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

class CycleGAN_PFL(CycleGANModel):
    """
    This class implements a CycleGAN model with perceptual feature loss, for learning image-to-image translation without paired data.
    To calculate the perceptual loss, a pretrained ResNet34 is used.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    Perceptual Loss paper: https://arxiv.org/abs/1706.09138
    """

    def name(self):
        return 'CycleGAN_Image_Perceptual_Loss'

    def __init__(self, opt):
        super().__init__(opt, num_dimensions=2)

        # Discriminators
        if self.isTrain:
            # The feature network is a pretrained ResNet wich is used to calculate the perceptual loss
            self.netFeat = networks.define_feature_network(opt.which_model_feat, self.gpu_ids)
                
        self.criterionFeat = mse_loss

        # Set loss weights
        if bool(self.opt.lambda_feat):
            self.lambda_feat_AfB = self.lambda_feat_BfA = self.lambda_feat_fArecB = self.lambda_feat_fBrecA = self.lambda_feat_ArecA = self.lambda_feat_BrecB = self.opt.lambda_feat
        else:
            self.lambda_feat_AfB = self.opt.lambda_feat_AfB    
            self.lambda_feat_BfA = self.opt.lambda_feat_BfA

            self.lambda_feat_fArecB = self.opt.lambda_feat_fArecB
            self.lambda_feat_fBrecA = self.opt.lambda_feat_fBrecA

            self.lambda_feat_ArecA = self.opt.lambda_feat_ArecA
            self.lambda_feat_BrecB = self.opt.lambda_feat_BrecB

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.\n
        input (dict): include the data itself and its metadata information.\n
        The option 'direction' can be used to swap domain A and domain B.
        """
        super().set_input(input)
        AtoB = self.opt.which_direction == 'AtoB'
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B

        # Feature loss
        self.feat_loss_AfB = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)) * self.lambda_feat_AfB    
        self.feat_loss_BfA = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.fake_A)) * self.lambda_feat_BfA

        self.feat_loss_fArecB = self.criterionFeat(self.netFeat(self.fake_A), self.netFeat(self.rec_B)) * self.lambda_feat_fArecB
        self.feat_loss_fBrecA = self.criterionFeat(self.netFeat(self.fake_B), self.netFeat(self.rec_A)) * self.lambda_feat_fBrecA

        self.feat_loss_ArecA = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.rec_A)) * self.lambda_feat_ArecA 
        self.feat_loss_BrecB = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.rec_B)) * self.lambda_feat_BrecB 

        self.feat_loss = self.feat_loss_AfB + self.feat_loss_BfA + self.feat_loss_fArecB + self.feat_loss_fBrecA + self.feat_loss_ArecA + self.feat_loss_BrecB

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.feat_loss
        self.loss_G.backward()

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

