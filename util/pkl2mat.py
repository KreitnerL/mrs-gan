import argparse
import pickle
from os import makedirs
from os.path import join, split, splitext

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from scipy.io import loadmat, savemat
from smoothing1D import smooth
from torch import load

'''
The CPU_Unpickler comes from code in a comment on the following
website:
https://github.com/pytorch/pytorch/issues/16797 - from mmcdermott
This was necessary because just adding map_location='cpu' was not
working on my machine.
'''


'''
In my experiments, I track the loss, MSE, R2, and Error for 
individual classes and groups of classes. This code plots that 
information in a useful format for me. It also gives flexibility
if you are only interested in particular metrics for a given 
experiment.

You should only have to modify the plotting if you want to use it.
If not, simply specify the input path and optionally the save dir.
'''

__all__ = ['pkl2mat']

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: load(BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def pkl2mat(input, args):
    with open(args.input, 'rb') as file:
        dict = CPU_Unpickler(file).load()
    savemat(args.save, mdict=dict, do_compression=True)

    if args.plot:
        # dict = loadmat(args.save)
        root, _ = split(args.save)
        train, val = join(root,'training'), join(root,'validation')
        try: makedirs(train) 
        except FileExistsError: pass
        try: makedirs(val) 
        except FileExistsError: pass
        if args.r2 or args.plot_all:            
            plot_metric(dict['training']['r2'],'training','r2',train,False)
            if args.smooth: plot_metric(dict['training']['r2'],'training','r2',train,15)
            plot_metric(dict['validation']['r2'],'validation','r2',val,False)
            if args.smooth: plot_metric(dict['validation']['r2'],'validation','r2',val,5)
        if args.mse or args.plot_all:            
            plot_metric(dict['training']['mse'],'training','mse',train,False)
            if args.smooth: plot_metric(dict['training']['mse'],'training','mse',train,15)
            plot_metric(dict['validation']['mse'],'validation','mse',val,False)
            if args.smooth: plot_metric(dict['validation']['mse'],'validation','mse',val,5)
        if args.error or args.plot_all:            
            plot_metric(dict['training']['error'],'training','error',train,False)
            if args.smooth: plot_metric(dict['training']['error'],'training','error',train,15)
            plot_metric(dict['validation']['error'],'validation','error',val,False)
            if args.smooth: plot_metric(dict['validation']['error'],'validation','error',val,5)                    
        if args.loss or args.plot_all:            
            plot_metric(dict['training']['ls_loss'],'training','ls_loss',train,False)
            plot_metric(dict['validation']['ls_loss'],'validation','ls_loss',val,False)


def plot_metric(dict,phase,label,path,sm=False):
    l = len(dict['Cho']) if not 'loss' in label else len(dict)
    if phase=='training':
        x = np.arange(0,l/16,1/16)
    elif phase=='validation':
        x = np.arange(1,l+1)
    basename = join(path,phase + '_' + label)


    if not label.lower()=='ls_loss':
        keys = dict.keys()
        ind = np.arange(0,len(keys))

        # Overall
        plt.figure(figsize=(20,15))
        for k in keys:
            f = -1. if label=='r2' else 1.
            if sm is False:
                y = f * np.asarray(dict[k])
            else:
                y = smooth(f * np.asarray(dict[k]),window_len=sm, window='flat')
            if 'overall' in k.lower():
                plt.plot(x,y[:len(x)],'-.',label=k)
            else:
                plt.plot(x,y[:len(x)],label=k)
        plt.legend(loc=0)
        if sm is False:
            filename = basename + '_overall_unsmoothed.png'
        else:
            filename = basename + '_overall_smoothed.png'
        plt.savefig(filename)
        

        # Metabolites
        plt.figure(figsize=(20,15))
        for i, k in zip(ind, keys):
            if i<7:
                f = -1 if label=='r2' else 1
                if sm is False:
                    y = f * np.asarray(dict[k])
                else:
                    y = smooth(np.squeeze(f * np.asarray(dict[k])),window_len=sm, window='flat')
                if 'overall' in k.lower():
                    plt.plot(x,y[:len(x)],'-.',label=k)
                else:
                    plt.plot(x,y[:len(x)],label=k)
        plt.legend(loc=0)
        if sm is False:
            filename = basename + '_per_metabolite_unsmoothed.png'
        else:
            filename = basename + '_per_metabolite_smoothed.png'
        plt.savefig(filename)


        # Categories
        plt.figure(figsize=(20,15))
        for i, k in zip(ind, keys):
            if i>=7:
                f = -1 if label=='r2' else 1
                if sm is False:
                    y = f * np.asarray(dict[k])
                else:
                    y = smooth(np.squeeze(f * np.asarray(dict[k])),window_len=sm, window='flat')
                if 'overall' in k.lower():
                    plt.plot(x,y[:len(x)],'-.',label=k)
                else:
                    plt.plot(x,y[:len(x)],label=k)
        plt.legend(loc=0)
        if sm is False:
            filename = basename + '_per_category_unsmoothed.png'
        else:
            filename = basename + '_per_category_smoothed.png'
        plt.savefig(filename)
    else:
        # Overall
        plt.figure(figsize=(20,15))
        y = np.squeeze(np.asarray(dict))
        plt.plot(x,y,label=label)
        plt.legend(loc=1)
        filename = basename + '_overall_unsmoothed.png'
        plt.savefig(filename)        

    plt.close('all')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input pickle file')
    parser.add_argument('--save',  type=str, default=None, help='save path')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--r2', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)
    parser.add_argument('--loss', action='store_true', default=False)
    parser.add_argument('--error', action='store_true', default=False)
    parser.add_argument('--smooth', action='store_true', default=False)
    parser.add_argument('--plot_all', action='store_true', default=False)    

    args = parser.parse_args()

    if args.input.__class__.__name__=='list':
        for file in args.input:
            if args.save is None:
                root, _ = splitext(file)
                args.save = root + '.mat'
            pkl2mat(file, args)

    elif args.input.__class__.__name__=='str':
        if args.save is None:
            root, _ = splitext(args.input)
            args.save = root + '.mat'
        pkl2mat(args.input, args)
