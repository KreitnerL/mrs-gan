
from torch.nn.modules.loss import L1Loss, MSELoss
from models.cycleGAN_WGP import CycleGAN_WGP


class CycleGAN_WGP_IFL(CycleGAN_WGP):

    def __init__(self, opt):
        super().__init__(opt)
        self.encoder_A, self.transformer_A, self.decoder_A = self.netG_A.get_components()
        self.encoder_B, self.transformer_B, self.decoder_B = self.netG_B.get_components()

    def name(self):
        return 'CycleGAN_WGP_IFL'

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        self.real_A = self.input_A
        feat_A = self.encoder_A(self.real_A)
        self.fake_B = self.decoder_A(self.transformer_A(feat_A))
        self.rec_A = self.netG_B(self.fake_B)
        self.identity_A = self.decoder_B(self.transformer_B(feat_A))

        if self.opt.phase != 'val':
            self.real_B = self.input_B
            feat_B = self.encoder_B(self.real_B)
            self.fake_A = self.decoder_B(self.transformer_B(feat_B))
            self.rec_B = self.netG_A(self.fake_A)
            self.identity_B = self.decoder_A(self.transformer_A(feat_B))

    def calculate_G_loss(self):
        loss = super().calculate_G_loss()
        if self.opt.lambda_feat > 0:
            loss += self.opt.lambda_feat*self.criterionIdt(self.identity_A, self.real_A)
            loss += self.opt.lambda_feat*self.criterionIdt(self.identity_B, self.real_B)
        return self.loss_G
