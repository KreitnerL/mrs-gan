from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()

    def initialize(self):
        BaseOptions.initialize(self)
        # visdom and HTML visualization parameters
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--plot_freq', type=int, default=1000, help="number of iterations between plotting")
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--val_path', type=str, default=None, help='File path to the pretrained random forest dump.')
        self.parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')

        # network saving and loading parameters
        self.parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        self.parser.add_argument('--n_epochs', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term beta1 of adam optimizer')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term beta2 of adam optimizer')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--n_critic', type=int, default=1, help='number of optimizations for the critic before generator is optimized')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        
        self.parser.add_argument('--TTUR', action='store_true', help='Enable the Two Time-scale Update Rule for stabilizing training and reducing the chance of mode collapse')
        self.parser.add_argument('--n_epochs_gen', type=int, default=100, help='# of iter at starting generator learning rate')
        self.parser.add_argument('--n_epochs_gen_decay', type=int, default=100, help='# of iter to linearly decay generator learning rate to zero')
        self.parser.add_argument('--n_epochs_dis', type=int, default=100, help='# of iter at starting discriminator learning rate')
        self.parser.add_argument('--n_epochs_dis_decay', type=int, default=100, help='# of iter to linearly decay discriminator learning rate to zero')
        self.parser.add_argument('--glr', type=float, default=0.0002, help='initial generator learning rate for adam')
        self.parser.add_argument('--dlr', type=float, default=0.0002, help='initial discriminator learning rate for adam')

        self.parser.add_argument('--gan_mode', type=str, default='vanilla', help='type of GAN loss [vanilla | lsgan | wasserstein]')
        self.parser.add_argument('--weight_norm', type=str, default='gp', help='Only used with gan_mode=wasserstein. Method used to enforce lipschitz continuity [clip | gp | sn]')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--lambda_feat', type=float, default=0, help='weight for perception loss (overwrite all)')
        # self.parser.add_argument('--lambda_feat_AfB', type=float, default=0, help='weight for perception loss between real A and fake B ')
        # self.parser.add_argument('--lambda_feat_BfA', type=float, default=0, help='weight for perception loss between real B and fake A ')
        # self.parser.add_argument('--lambda_feat_fArecB', type=float, default=0, help='weight for perception loss between fake A and reconstructed B ')
        # self.parser.add_argument('--lambda_feat_fBrecA', type=float, default=0, help='weight for perception loss between fake B and reconstructed A ')
        # self.parser.add_argument('--lambda_feat_ArecA', type=float, default=0, help='weight for perception loss between real A and reconstructed A ')
        # self.parser.add_argument('--lambda_feat_BrecB', type=float, default=0, help='weight for perception loss between real B and reconstruced B ')
        self.parser.add_argument('--lambda_entropy', type=float, default=0, help='weight for entropy loss')

        
        self.isTrain = True
