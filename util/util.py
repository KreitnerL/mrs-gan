from __future__ import print_function
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import io
import cv2
import matplotlib.pyplot as plt
import sys
import re
from scipy.stats import pearsonr

fig = ax = None


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# define a function which returns an image as numpy array from figure
# TODO add global option
def get_img_from_fig(x, y, xlabel='', ylabel='', dpi=180, magnitude=True):
    global fig, ax
    if fig==None:
        fig, ax = plt.subplots()
        # fig.set_size_inches(500/dpi, 350/dpi, forward=True)
    else:
        plt.figure(fig.number)
    if magnitude:
        y = y.sum(-2, True)
    for channel in reversed(range(y.size(-2))):
        y_i = y.select(-2, channel)
        ax.plot(x,y_i.squeeze().detach().cpu().numpy())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.cla()

    return img


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def progressbar(it, prefix="", size=60, file=sys.stdout, num_iters = float("inf")):
    count = min(len(it), num_iters)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        if i>=num_iters:
            break
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def is_set_of_type(dir, type):
    """
    Checks if there is at least one .mat file under the given path (including subfolders)
    """
    for _, _, fnames in os.walk(dir):
        if any(fname.endswith(type) for fname in fnames):
            return True
    return False

def smooth_kernel(x, kernel_size=5):
    """
    Smoothes the given graph by sliding a kernel along each datapoint that takes the average of the selected values.

    Parameters:
    ----------
        - x (list): List of length L
        - kernel_size (int): Size of the kernel. Default = 5

    Returns:
    -------
        - List of length L-kernelsize+1
    """
    x_smooth = []
    for i in range(len(x)-kernel_size):
        x_smooth.append(np.mean(x[i:i+kernel_size]))
    return x_smooth

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    Taken from
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters
    ----------
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns
    -------
        the smoothed signal
        
    Example
    --------

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len or window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def load_loss_from_file(opt, path):
    """
    Loads the given loss file, extracts all losses and returns them in a struct
    """
    legend = []
    y = []
    has_legend = False
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        if line.startswith('='):
            continue
        # if int(re.search(r'\d+', line).group()) >= opt.epoch_count:
        #     break
        line =  re.sub('\(.*\)', '', line)
        y_i = []
        for t in line.split():
            try:
                y_i.append(float(t))
            except ValueError:
                if not has_legend and len(t)>1:
                    legend.append(t.replace(':', ''))
        y.append(y_i)
        has_legend=True
    y.pop(0)
    x = [*np.array(range(len(y)))/10000]

    return {'X': x, 'Y': y, 'legend': legend}

def normalize(spectra: np.ndarray) -> np.ndarray:
    """
    Normalizes the given set of spectra to [-1,1].

    Parameters
    ---------
        - spectra: Numpy array of shape NxCxL containing N spectra of length L

    Returns
    -------
        - Numpy array of Shape NxCxL containing the normalized spectra
    """
    max_per_spectrum = np.amax(abs(spectra),(1,2))
    max_per_spectrum = np.repeat(max_per_spectrum[:, np.newaxis], spectra.shape[1], axis=1)
    max_per_spectrum = np.repeat(max_per_spectrum[:, :, np.newaxis], spectra.shape[2], axis=2)
    return np.divide(spectra, max_per_spectrum)

def compute_error(predictions: list, y):
        """
        Compute the realtive errors and the average per metabolite

        Parameters
        ---------
        - predictions: List of predicted quantifications by the random forest
        - y: List of quantifications. N2xM, M = number of metabolites, N2 = number of spectra
        
        Returns
        -------
        - err_rel: List of relative errors. M x N2
        - avg_err_rel: Average relative error per metabolite. M x 1
        """
        err_rel = []
        avg_err_rel = []
        pearson_coefficient = []
        for metabolite in range(len(y[0])):
            err_rel.append((abs(predictions[:,metabolite] - y[:,metabolite])) / (abs(y[:,metabolite])))
            avg_err_rel.append(np.mean(err_rel[metabolite]))
            pearson_coefficient.append(abs(pearsonr(predictions[:,metabolite], y[:,metabolite])[0]))
        
        return err_rel, avg_err_rel, pearson_coefficient


def save_boxplot(err_rel, avg_err_rel, path: str, labels: list):
    """
    Save a boxplot from the given relative errors.

    Parameters
    ---------
    - err_rel: List of relative errors. M x N2
    - path: directory where the plot should be saved.
    """
    global fig
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    max_y = 0.15 if max(np.array(avg_err_rel)) < 0.1 else 1.0
    plt.boxplot(err_rel, notch = True, labels=labels, showmeans=True, meanline=True)
    plt.ylabel('Relative Error')
    plt.title('Error per predicted metabolite')
    plt.ylim([0,max_y])
    path = path+'_rel_err_boxplot.png'
    plt.savefig(path, format='png')
    plt.cla()
    print('Saved error plot at', path)

# def load_options(path):
#     with open(path) as file:
#         opt = dict()
#         for line in file:
#             line = line.rstrip()
#             if line.startswith('-'):
#                 continue
#             args = line.split(': ')
#             try:
#                 opt[args[0]] = eval(args[1])
#             except:
#                 opt[args[0]] = args[1]
#     return opt

def update_options(opt: Namespace, update_opts: dict):
    opt = vars(opt)
    opt.update(update_opts)
    opt = Namespace(**opt)
    return opt

