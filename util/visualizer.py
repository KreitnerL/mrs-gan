from util.html import HTML
import numpy as np
import os
import ntpath
import time
from . import util
import matplotlib.pyplot as plt
from util.util import load_validation_from_file, smooth_kernel, load_loss_from_file

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        if opt.isTrain:
            # create a logging file to store training losses
            self.loss_log = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            self.validation_log = os.path.join(opt.checkpoints_dir, opt.name, 'validation.txt')
            if opt.continue_train:
                if os.path.isfile(self.loss_log):
                    self.plot_data = load_loss_from_file(self.loss_log)
                    if len(self.plot_data['legend']) == 0:
                        del self.plot_data
                    print('Loaded loss from', self.loss_log)
                if os.path.isfile(self.validation_log):
                    self.validation_score = load_validation_from_file(self.validation_log)
                    print('Loaded validation scores from', self.validation_log)
            elif os.path.isfile(self.loss_log):
                # Erase old content
                open(self.loss_log, 'w').close()
                open(self.validation_log, 'w').close()

            with open(self.loss_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def plot_current_losses(self):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        else:
            plt.figure(self.figure.number)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(self.name + ' loss over time')
        plt.yscale('symlog')
        plt.ylim((-50,80))
        x = self.plot_data['X']
        y = np.array(self.plot_data['Y']).transpose()
        for i, loss in enumerate(y):
            if i>=3:
                break
            plt.plot(x, loss, label=self.plot_data['legend'][i])
        plt.legend()

        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'loss.png')
        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.cla()

    def plot_current_validation_score(self, score, total_iters):
        with open(self.validation_log, 'a') as f:
            f.write(', '.join(map(str, score))+'\n')

        if not hasattr(self, 'validation_score'):
            self.validation_score = []
        self.validation_score.append(score)
        if not hasattr(self, 'figure2'):
            self.figure2 = plt.figure()
        else:
            plt.figure(self.figure2.number)

        plt.xlabel('epoch')
        plt.ylabel('R-score')
        plt.title(self.name + ' validation score over time')
        plt.ylim([0,1])
        step_size = int(total_iters/len(self.validation_score))
        x = list(range(step_size, total_iters+1, step_size))
        for i in range(len(score)):
            plt.plot(x, np.array(self.validation_score)[:,i])

        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'validation_score.png')
        plt.savefig(path, format='png')
        plt.cla()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, iter):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(iter)
        self.plot_data['Y'].append([losses[k].cpu().data.numpy() for k in self.plot_data['legend']])

        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.loss_log, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def save_smooth_loss(self):
        """Stores the current loss as a png image.
        """
        num_points = len(self.plot_data['Y'])

        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        else:
            plt.figure(self.figure.number)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(self.name + ' loss over time')
        plt.yscale('symlog')
        plt.ylim((-50,80))
        x = self.plot_data['X']
        y_all = np.array(self.plot_data['Y']).transpose()
        y = []
        for y_i in y_all:
            y.append(smooth_kernel(y_i))
        x = np.linspace(x[0],x[-1],len(y[0]))
        for i, loss in enumerate(y):
            plt.plot(x, loss, label=self.plot_data['legend'][i])
        plt.legend()

        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'loss_smooth.png')
        plt.savefig(path, format='png')
        plt.cla()

def save_images(webpage: HTML, visuals: dict, image_path: list, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im in visuals.items():
        if im is None:
            continue
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)
