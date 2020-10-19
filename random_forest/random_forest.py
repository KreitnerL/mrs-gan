# Script to train a random forest model to quantify the given number of metabolites of an MR spectrum.
# The model is trained on the given training set and validated on a given validation set.

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import os

class RandomForest:
    """
    Create a multi regression random forest for the given labels.

    Parameters
    ---------
    - num_trees: Number of decision trees used in the random forest.
    - labels: List of metabolite names.
    """
    def __init__(self, num_trees, labels):
        # set random state to 0 for reproducability
        self.num_trees = num_trees
        self.labels = labels
        self.regressor = RandomForestRegressor(n_estimators=num_trees, random_state=0)

    def train(self, x, y):
        """
        Train the random forest on the given dataset.
        Parameters
        ---------
        - x: List of spectra. N1 x L double, N1 = number of spectra, L length of spectra
        - y: List of quantifications. M x N2, M = number of metabolites, N2 = number of spectra
        """
        print('training random forest with {0} trees on {1} data samples', self.num_trees, len(x))
        self.regressor.fit(x, y)

    def test(self, x) -> list:
        """
        Test the random forest on the given dataset.
        Parameters
        ---------
        - x: List of spectra. N1 x L double, N1 = number of spectra, L length of spectra

        Returns
        -------
        prediction : The predicted values. N2 x M
        """
        return self.regressor.predict(x)

    def compute_error(self, predictions: list, y):
        """
        Compute the realtive errors and the average per metabolite

        Parameters
        ---------
        - predictions: List of predicted quantifications by the random forest
        - y: List of quantifications. M x N2, M = number of metabolites, N2 = number of spectra
        
        Returns
        -------
        - err_rel: List of relative errors. M x N2
        - avg_err_rel: Average relative error per metabolite. M x 1
        """
        err_rel = []
        avg_err_rel = []
        for metabolite in len(self.labels):
            err_rel[metabolite] = (abs(predictions[metabolite] - y[metabolite])) / (abs(y[metabolite]))
            avg_err_rel[metabolite] = np.mean(err_rel[metabolite], axis=2)
            print('Relative error {0}: {1}\n', self.labels(metabolite), avg_err_rel[metabolite])
        return err_rel, avg_err_rel

    def save_plot(self, err_rel, path: str):
        """
        Save a boxplot from the given relative errors.

        Parameters
        ---------
        - err_rel: List of relative errors. M x N2
        - path: directory where the plot should be saved.
        """
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        else:
            plt.figure(self.figure.number)
        plt.boxplot(err_rel, notch = True, labels=self.labels)
        plt.ylabel('Relative Error')
        plt.title('Error per predicted metabolite')
        path = os.path.join(path, 'rel_err_boxplot.png')
        plt.savefig(path, format='png')
        plt.cla()

def train_val(x_train, x_test, y_train, y_test, labels, path, num_trees=100):
    """
    Performs training and validation for the given dataset
    """
    rf = RandomForest(num_trees, labels)
    rf.train(x_train, y_train)
    predictions = rf.test(x_test)
    err_rel, avg_err_rel = rf.compute_error(predictions, y_test)
    rf.save_plot(err_rel, path)