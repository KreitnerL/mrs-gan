# Options

This folder contains files specifying the available options during [training](#train-options) and [testing](#test-options). It also contains the options for creating the dataset.

All available options are listed below:

## Base options:
- `--dataroot`: (**REQUIRED**) path to images (should have subfolders trainA, trainB, valA, valB, etc). type=str
- `--gpu_ids`: gpu ids to run the code on: e.g. 0;  0,1,2; 0,2. use -1 for CPU'. To profit from multiple GPUs choose a big `batch_size`. type=str, default='0'
- `--batch_size`: input batch size. type=int, default=1
- `--name`: name of the experiment. It decides where to store samples and models. type=str, default='experiment_name'
- `--checkpoints_dir`: model checkpoints are saved here. type=str, default='./checkpoints'
- `--dataset_mode`: chooses how datasets are loaded. [unaligned | LabeledMatSpectralDataset]'. type=str, default='unaligned'
- `--model`: chooses which model to use. [cycle_gan]. type=str, default='cycle_gan'
- `--epoch_count`: which epoch to load? set to latest to use latest cached model. type=str, default='latest'
- `--nThreads`: # threads for loading data, default=2. type=int
- `--input_nc`: # of input image channels. type=int, default=3
- `--output_nc`: # of output image channels. type=int, default=3
- `--loadSize`: scale images to this size. type=int, default=286
- `--fineSize`: after scaling, crop image to this size. type=int, default=256
- `--ngf`: # of gen filters in first conv layer. type=int, default=64
- `--ndf`: # of discrim filters in first conv layer. type=int, default=64
- `--which_model_netD`: selects model to use for netD. type=str, default='basic'
- `--which_model_netG`: selects model to use for netG. type=str, default='resnet_9blocks'
- `--which_model_feat`: selects model to use for feature network. type=str, default='resnet34'
- `--n_layers_D`: , only used if which_model_netD==n_layers. type=int, default=3
- `--norm`: instance normalization or batch normalization [instance | batch]. type=str, default='instance'
- `--shuffle`: if false, takes images in order to make batches, otherwise takes them randomly, default=True, action='store_true'
- `--identity`: use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1'. type=float, default=0.0
- `--max_dataset_size`: Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded. type=int, default=float("inf")
- `--resize_or_crop`:  scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]. type=str, default='resize_and_crop'
- `--no_flip`, if specified, do not flip the images for data augmentation. action='store_true`, type=bool, default=False

Specifically for spectra:
- `--split`: split the dataset into training, validating, and testing segments. action='store_true', default=False
- `--val_split`: percent of data to use for validation. type=float, default=0.2
- `--test_split`: percent of data to use for testing. type=float, default=0.0
- `--phase_data_path`: if data has already been split, indicate the path for the data split index, type=str
- `--CpoolSize`: critic value pool size for relative loss. type=int, default=50
- `--input_dim`: dimension of input data - "1" for spectra, "2" for images,"3" for image volumes, type=int, default=1
- `--real`: Use only the real portion of the signal. action="store_false", default=True
- `--imag`: Use only the real portion of the signal. action="store_false", default=True
- `--shuffle_data`: Shuffle sequence of data when initially extracted from dataset. action='store_true', default=True
- `--normalize`: Normalize the input data, action='store_true', default=False
- `--standardize`: Standardize the input data. action='store_true', default=False
- `--norm_range`: Range in which the input data should be normalized. type=list, default=[-1, 1]
- `--pad_data`: Pad data when loading. Most ResNet architectures require padding MRS data by 21. type=int, default=0


For visuals:
- `--quiet`: Does not print the options in the terminal when initializing. action='store_true', default=False
- `--plot_grads`: Plot the gradients for each network after the backward step, action='store_true', default=False
- `--display_winsize`: display window size. type=int, default=256
- `--display_id`: window id of the web display. type=int, default=1
- `--display_port`: visdom port of the web display. type=int, default=8097
- `--display_single_pane_ncols`: if positive, display all images in a single visdom web panel with certain number of images per row. type=int, default=0

<a name="train-options"></a>

## Train options
- `--save_epoch_freq`: frequency of saving checkpoints at the end of epochs. type=int, default=5
- `--continue_train`: continue training: load the latest model. action='store_true'
- `--phase`: train, val, test, etc. type=str, default='train'
- `--n_epochs`: # of iter at starting learning rate. type=int, default=100
- `--n_epochs_decay`: # of iter to linearly decay learning rate to zero. type=int, default=100
- `--beta1`: momentum term of adam, type=float, default=0.5
- `--lr`: initial learning rate for adam, type=float, default=0.0002
- `--gan_mode`: TODO
- `--lambda_A`: weight for cycle loss (A -> B -> A). type=float, default=10.0
- `--lambda_B`: weight for cycle loss (B -> A -> B). type=float, default=10.0
- `--lambda_feat_AfB`: weight for perception loss between real A and fake B. type=float, default=0
- `--lambda_feat_BfA`: weight for perception loss between real B and fake A. type=float, default=0
- `--lambda_feat_fArecB`: weight for perception loss between fake A and reconstructed B, type=float, default=0
- `--lambda_feat_fBrecA`: weight for perception loss between fake B and reconstructed A. type=float, default=0
- `--lambda_feat_ArecA`: weight for perception loss between real A and reconstructed A. type=float, default=0
- `--lambda_feat_BrecB`: weight for perception loss between real B and reconstruced B. type=float, default=0
- `--pool_size`: the size of image buffer that stores previously generated images. type=int, default=50

Specifically for spectra:
- `--file_ext`: can add additional information to select specific files from the dataset. type=str, default='proc.npz'
- `--input_ext`: can add additional information to select specific files from the dataset. type=str, default='proc.npz'
- `--folder_ext`: can add additional information to select specific folders from the dataset. type=str, default='UCSF'
- `--k_folds`: number of folds for a cross-validation training scheme. type=int, default=-1

For visuals:
- `--display_freq`: frequency of showing training results on screen. type=int, default=100
- `--print_freq`: frequency of showing training results on console. type=int, default=100
- `--save_latest_freq`: frequency of saving the latest results. type=int, default=5000
- `--no_html`: do not save intermediate training results to `[opt.checkpoints_dir]/[opt.name]/web/` action='store_true'

<a name="test-options"></a>

## Test options
- `--results_dir`: saves results here. type=str, default='./results/'
- `--ntest`: # of test examples.. type=int, default=float("inf")
- `--aspect_ratio`: aspect ratio of result images. type=float, default=1.0
- `--phase`: train, val, test, etc. type=str, default='test'
- `--how_many`: how many test images to run. type=int, default=50


## Dataset Creation
- `--source_dir_A`, Directory of the dataset of domain A, type=str
- `--source_dir_B`, Directory of the dataset of domain B, type=str
- `--save_dir`, Directory where the dataset will be saved, type=str
- `--file_ext_spectra`, File extension of the processed spectra DICOM files, type=str, default='proc.dcm'
- `--file_ext_metabolic_map`, File extension of the metabolic map, type=str, default='NAA.dcm'
- `--force`, If true, overwrites all exisiting .npz, .mat, .dat files, type=bool, default=False

- `--normalize`, normalize the spectra in preprocessing, type=bool, default=False
- `--standardize`, standardize the spectra in preprocessing, type=bool, default=False
- `--pad_data`, pad_data the spectra in preprocessing, type=bool, default=False
- `--input_nc`, number if input channels, type=int, default=2
- `--real`, only use real part of the spectra, type=bool, default=False
- `--imag`, only use imaginary part of the spectra, type=bool, default=False
- `--split`, Split the data into train, validation and test set, type=bool, default=True
- `--val_split`, Part of dataset that is used for validation, type=float, default=0.3
- `--test_split`, Part of dataset that is used for testing, type=float, default=0
- `--shuffle_data`, Select spectra for training / testing randomly, type=bool, default=True
- `--quiet`, Does not print the options in the terminal when initializing, action='store_true', default=False
