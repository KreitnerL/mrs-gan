# Script to train a random forest model to quantify the given number of metabolites of an MR spectrum.
# The model is trained on the given training set and validated on a given validation set.

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
from joblib import dump, load
import os

class RandomForest:
    """
    Create a multi regression random forest for the given labels.

    Parameters
    ---------
    - num_trees: Number of decision trees used in the random forest.
    - labels: List of metabolite names.
    """
    def __init__(self, num_trees, labels, load_from=None):
        # set random state to 0 for reproducability
        self.num_trees = num_trees
        self.labels = labels
        self.load_from = load_from+ '.joblib'
        if load_from and os.path.isfile( self.load_from):
            self.load( self.load_from)
            self.pretrained = True
        else:
            self.regressor = RandomForestRegressor(n_estimators=num_trees, random_state=0)
            self.pretrained = False

    def train(self, x, y):
        """
        Train the random forest on the given dataset.
        Parameters
        ---------
        - x: List of spectra. N1 x L double, N1 = number of spectra, L length of spectra
        - y: List of quantifications. M x N2, M = number of metabolites, N2 = number of spectra
        """
        print('training random forest with {0} trees on {1} data samples...'.format(self.num_trees, len(x)))
        start = time.time()
        self.regressor.fit(x, y)
        print('training completed in {:.3f} sec'.format(time.time()-start))
        self.save(self.load_from)
            

    def save(self, filepath):
        print('storing random forest weights at', filepath)
        dump(self.regressor, filepath)

    def load(self, filepath):
        print('Loading pretrained model from', filepath)
        self.regressor = load(filepath)


    def predict(self, x) -> list:
        """
        Test the random forest on the given dataset.
        Parameters
        ---------
        - x: List of spectra. N1 x L double, N1 = number of spectra, L length of spectra

        Returns
        -------
        prediction : The predicted values. N2 x M
        """
        print('Predicting values for', len(x), 'samples')
        prediction =  self.regressor.predict(x)
        return np.array(prediction)