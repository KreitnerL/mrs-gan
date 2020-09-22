# Options that are used specifically to configure training.
# If an options is not set, its default will be used.


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--file_ext', type=str, default='proc.npz', help='can add additional information to select specific files from the dataset')
        self.parser.add_argument('--input_ext', type=str, default='proc.npz', help='can add additional information to select specific files from the dataset')
        self.parser.add_argument('--folder_ext', type=str, default='UCSF', help='can add additional information to select specific folders from the dataset')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=25000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--starting_epoch', type=int, default=1, help='when training, or continuing training, which epoch should start first?')
        self.parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=20, help='# of iter to linearly decay learning rate to zero')
        # self.parser.add_argument('--niter_gen', type=int, default=20, help='# of iter at starting generator learning rate')
        # self.parser.add_argument('--niter_gen_decay', type=int, default=20, help='# of iter to linearly decay generator learning rate to zero')
        # self.parser.add_argument('--niter_dis', type=int, default=20, help='# of iter at starting discriminator learning rate')
        # self.parser.add_argument('--niter_dis_decay', type=int, default=20, help='# of iter to linearly decay discriminator learning rate to zero')
        # self.parser.add_argument('--no_TTUR', action='store_true', help='Disable the Two Time-scale Update Rule for stabilizing training and reducing the chance of mode collapse')
        # self.parser.add_argument('--beta', type=float, default=0.500, help='momentum term of adam')
        self.parser.add_argument('--beta1', type=float, default=0.900, help='momentum term of adam for generator')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam for discriminator')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')#required='no_TTUR',
        # self.parser.add_argument('--glr', type=float, default=0.0002, help='initial generator learning rate for adam')
        # self.parser.add_argument('--dlr', type=float, default=0.0002, help='initial discriminator learning rate for adam')
        # self.parser.add_argument('--timestep_ratio', type=float, default=5, help='TTUR ratio of dlr/glr, should be >1') # default=0.5
        # self.parser.add_argument('--label_smoothing', type=float, help='default = 0.2: smoothed target label to help stabilize training. Recommended starting value is 0.9')#default=0.2,
        # self.parser.add_argument('--D_guidance', action="store_true", help='Activate additional discriminator loss to guide it towards class labels')#default=0.2,

        # self.parser.add_argument('--critic_iters', type=float, default=5, help='number of times to update the discriminator before updating the generator')
        # self.parser.add_argument('--gen_timestep', type=float, default=0.4, help='TTUR generator update step, should be <1') # default=0.5
        # self.parser.add_argument('--dis_timestep', type=float, default=2.0, help='TTUR discriminator update step, should be >1')
        # self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        # Loss Options
        # # GAN Loss
        # self.parser.add_argument('--lambda_ganloss', type=float, default=1.0, help='weight for the GANLoss function')
        # # # Relativistic Avg Loss
        # self.parser.add_argument('--lambda_dCriticR', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
        # self.parser.add_argument('--lambda_dCriticF', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
        # self.parser.add_argument('--lambda_relAG', type=float, default=[1.0], help='weight for relativistic average generator loss')
        # self.parser.add_argument('--lambda_relAD', type=float, default=[1.0], help='weight for relativistic average discriminator loss')
        # # # Gradient Penalty
        # self.parser.add_argument('--lambda_gp', type=float, default=10.0, help='gradient penalty loss: weight for cycle loss (A -> B) discriminator')
        # self.parser.add_argument('--scaled_lambda_gp', action='store_true', help='Scales the gradient penalty according to the Wasserstein Distance')
        # self.parser.add_argument('--one_sided_gp', action='store_true', help='One-sided gradient penalty term with lower bound = 0')
        # # # Wasserstein
        # self.parser.add_argument('--wasserstein', type=bool, default=True, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.') # default=10
        # self.parser.add_argument('--critic_iters', type=int, default=5, help='number of inner iterations for the critic per outer iteration')
        # self.parser.add_argument('--grad_penalty', type=bool, default=True, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.') # default=10
        # self.parser.add_argument('--lambda_wasserstein', type=float, default=1.0, help='weight for the Wasserstein compenent of the loss function')
        # # # Perceptual Loss
        # self.parser.add_argument('--perception', action='store_true', default=False, help='Use perceptual feature loss when training')
        # self.parser.add_argument('--lambda_feat_loss', type=float, default=5.0, help='weight for the perceptual feature loss component')
        # self.parser.add_argument('--implement_perception', type=int, default=-1, help='Number of epochs before implementing the perception loss')
        # # Identity Loss
        self.parser.add_argument('--idt', action='store_true', default=False, help='Calculate an identity loss for the generator')
        self.parser.add_argument('--lambda_idt', type=float, default=2, help='weight for the identity loss in the generator')
        # self.parser.add_argument('--implement_idt', type=int, default=-1, help='Number of epochs before implementing the identity loss')
        self.parser.add_argument('--mse_loss', action='store_true', help='Use MSE loss to train the autoencoder')
        self.parser.add_argument('--lambda_mse', type=float, default=1, help='Lambda coefficient for MSE loss')
        # # Relative & Profile Entropy Loss
        self.parser.add_argument('--entropy', action='store_true', default=False, help='Calculate an entropy loss for the generator')
        self.parser.add_argument('--h', type=str, default='sel', help='"sel" for squared entropy loss or "el" for general entropy loss')
        # self.parser.add_argument('--implement_entropy', type=int, default=-1, help='Number of epochs before implementing the entropy loss')
        self.parser.add_argument('--lambda_h', type=float, default=5, help='weight for the entropy profile loss (KL Divergence)')
        self.parser.add_argument('--runningH', action='store_true', default=False, help='Calculate the entropy across the spectra vs of the entire spectra')
        self.parser.add_argument('--rHkernel', type=float, default=32, help='Size of the kernel for calculating the running entropy')
        self.parser.add_argument('--rh', type=str, default='rsel', help='"rsel" for relative squared entropy loss or "rel" for relative entropy loss')
        self.parser.add_argument('--lambda_rh', type=float, default=5, help='weight for the relative entropy loss')

        # self.parser.add_argument('--pool_size', type=int, default=25, help='the size of image buffer that stores previously generated images')
        # self.parser.add_argument('--no_html', action='store_true', default=True, help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # Autoencoder
        # self.parser.add_argument('--idt_loss', action='store_true', help='Use L1 identity loss to train the autoencoder')
        # self.parser.add_argument('--lambda_idt', type=float, default=1, help='Lambda coefficient for L1 identity loss')
        # self.parser.add_argument('--mse_loss', action='store_true', help='Use MSE loss to train the autoencoder')
        # self.parser.add_argument('--lambda_mse', type=float, default=1, help='Lambda coefficient for MSE loss')
        self.parser.add_argument('--k_folds', type=int, default=-1, help='number of folds for a cross-validation training scheme')
        # self.parser.add_argument('--pad_8', action='store_true')

        self.isTrain = True


# Todo: add growth_rate for dense architectures
