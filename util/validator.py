from models.cycleGAN import CycleGAN

from torch.utils.data.dataloader import DataLoader
from util.util import compute_error
import torch
import numpy as np
import sys


class Validator:
    """
    The Validator can validate a given cycleGAN model end to end by running a pretrained random forest on the generated fakes and computing the average relative error. 
    """
    def __init__(self, opt):
        self.opt = opt

    def get_validation_score(self, model: CycleGAN, dataset: DataLoader, num_batches=sys.maxsize):
        """
        Computes various validation metrics for the given model.

        Parameters
        ----------
            - model (CycleGAN): The current CycleGAN model
            - dataset (DataLoader): The dataset the validation samples should be taken from. If none is given, the configured validation set will be used. Default=None

        Returns
        -------
            - The Mean Absolute Error (L1) per metabolite. (M) with M=number of metabolites, N=number of test samples
            - The relative error per metabolite per fake. (MxN) with M=number of metabolites, N=number of test samples
            - The Average Relative Error per metabolite. (M) with M=number of metabolites
            - The Coefficient of Determination (R^2) pre metabolite. (M) with M=number of metabolites
        """
        predictions = []
        labels = []
        for i, data in enumerate(dataset):
            if i>num_batches:
                break
            model.set_input(data)  # unpack data from data loader
            labels.append(data['label_A'])
            model.test()           # run inference
            prediction = model.get_prediction()
            predictions.append(prediction)
        predictions = np.concatenate(predictions)
        labels = torch.cat(labels).numpy()
        avg_abs_err, err_rel, avg_err_rel, r2, median_rel_err = compute_error(predictions, labels)

        return avg_abs_err, err_rel, avg_err_rel, r2, median_rel_err