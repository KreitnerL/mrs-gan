
from torch.nn.modules.loss import L1Loss, MSELoss
from models.cycleGAN_WGP import CycleGAN_WGP


class CycleGAN_WGP_IFL(CycleGAN_WGP):

    def __init__(self, opt):
        super().__init__(opt)
        self.encoder_A, self.transformer_A, self.decoder_A = self.netG_A.get_components()
        self.encoder_B, self.transformer_B, self.decoder_B = self.netG_B.get_components()
        self.vertical_loss = lambda *args: self.opt.lambda_feat * MSELoss()(*args)
        self.horizontal_loss = lambda *args: 0.5*self.opt.lambda_feat * L1Loss()(*args)

    def name(self):
        return 'CycleGAN_WGP_IFL'

    def forward(self):
        """
        Uses Generators to generate fake and reconstructed spectra
        """
        if self.opt.phase != 'val' or self.opt.AtoB:
            self.real_A = self.input_A
            self.A_feat_1 = self.encoder_A(self.real_A)
            self.B_feat_1 = self.transformer_A(self.A_feat_1)

            self.fake_B = self.decoder_A(self.B_feat_1)

            self.B_feat_2 = self.encoder_B(self.fake_B)
            self.A_feat_2 = self.transformer_B(self.B_feat_2)
            self.rec_A = self.decoder_B(self.A_feat_2)

        if self.opt.phase != 'val' or not self.opt.AtoB:
            self.real_B = self.input_B
            self.B_feat_3 = self.encoder_B(self.real_B)
            self.A_feat_3 = self.transformer_B(self.B_feat_3)

            self.fake_A = self.decoder_B(self.A_feat_3)

            self.A_feat_4 = self.encoder_A(self.fake_A)
            self.B_feat_4 = self.transformer_A(self.A_feat_4)
            self.rec_B = self.decoder_A(self.B_feat_4)

    def calculate_G_loss(self):
        loss = super().calculate_G_loss()
        if self.opt.lambda_feat > 0:
            loss += self.vertical_loss(self.A_feat_1, self.A_feat_2)
            loss += self.vertical_loss(self.A_feat_3, self.A_feat_4)
            loss += self.vertical_loss(self.B_feat_1, self.B_feat_2)
            loss += self.vertical_loss(self.B_feat_3, self.B_feat_4)

            loss += self.horizontal_loss(self.A_feat_1, self.B_feat_1)
            loss += self.horizontal_loss(self.A_feat_2, self.B_feat_2)
            loss += self.horizontal_loss(self.A_feat_3, self.B_feat_3)
            loss += self.horizontal_loss(self.A_feat_4, self.B_feat_4)
        return self.loss_G
