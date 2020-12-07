import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from torch.utils.data.dataloader import DataLoader
from util.util import progressbar
from validation_networks.MLP.MLP_dataset import MLPDataset

class MLP(nn.Module):
    def __init__(self, save_path:str, val_fun=None, gpu: int = None, in_out = (1,1), num_neurons = (100, 100, 100), lr=0.001, batch_size: int = 1, num_epoch: int = 15, validate_every: int = 5000):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        num_neurons = [in_out[0], *num_neurons, in_out[1]]
        for i in range(1, len(num_neurons)):
            self.layers.add_module('Linear'+str(i), nn.Linear(num_neurons[i-1], num_neurons[i]))
            if i < len(num_neurons)-1:
                self.layers.add_module('ReLU'+str(i), nn.ReLU())
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr, betas=(0.9, 0.999))
        self.loss_fn = nn.MSELoss()
        self.save_path = save_path
        self.batch_size = batch_size
        self.validate_every = validate_every
        self.num_epoch = num_epoch
        self.val_fun = val_fun
        self.gpu=gpu
        self.load(self.save_path+'.pth')

        self.input = torch.Tensor(batch_size, 1, 1)
        if gpu is not None:
            torch.cuda.set_device(gpu)
            self.input = torch.cuda.FloatTensor(batch_size, 1, 1)
            self.label = torch.cuda.FloatTensor(batch_size, 1, 1)
            self.cuda()
        else:
            raise ValueError('MLP should run on a GPU!')

    def load(self, path):
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
            self.pretrained = True
            print('Loaded pretrained model from', path)
        else:
            self.pretrained = False

    def forward(self, x):
        self.input.resize_(x.size()).copy_(x)
        out = self.layers(self.input)
        return out

    def backward(self, pred, label):
        self.label.resize_(label.size()).copy_(label)
        loss = self.loss_fn(pred, self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        dataset = DataLoader(MLPDataset(x, None), batch_size=self.batch_size,
                             num_workers=0, drop_last=False, shuffle=False)
        return self._predict(dataset)

    def _predict(self, val_dataset: DataLoader):
        pred = []
        for spectra, _ in val_dataset:
            pred.append(self.forward(spectra).detach().cpu().numpy())

        return np.concatenate(pred)
        

    def train(self, spectra_train, labels_train, spectra_test, labels_test, tol: float = 1e-3, n_iter_no_change: int = 3):
        dataset_train = DataLoader(MLPDataset(spectra_train, labels_train),
                                    batch_size=self.batch_size,
                                    num_workers=0,
                                    shuffle=True,
                                    drop_last=False)
        dataset_test = DataLoader(MLPDataset(spectra_test, labels_test),
                                    batch_size=self.batch_size,
                                    num_workers=0,
                                    shuffle=False,
                                    drop_last=False)

        total_iters = 0
        loss = []
        val_range = slice(-n_iter_no_change, None)
        score = None
        early_stop = False
        early_stop_cond = lambda score: np.all((score.max(0) - score.min(0) < tol))

        for epoch in progressbar(range(self.num_epoch)):
            worst_batch = None
            worst_loss = 0
            if early_stop:
                break
            for (spectra, labels) in dataset_train:
                total_iters += self.batch_size
                pred = self.forward(spectra)
                loss.append(self.backward(pred, labels))
                pred = pred.detach().cpu().numpy()
                if loss[-1] > worst_loss:
                    worst_loss = loss[-1]
                    worst_batch = (spectra, labels, pred)

                if total_iters%self.validate_every==0:
                    self.plot_loss(loss)

                    pred = self._predict(dataset_test)
                    if score is None:
                        score = [self.val_fun(pred, labels_test)]
                    else:
                        score = np.append(score, [self.val_fun(pred, labels_test)], axis=0)
                        self.plot_val_score(score)
                    
                    if len(score)>n_iter_no_change and early_stop_cond(score[val_range]):
                        early_stop=True
                        break
            for _ in range(0):
                pred = self.forward(worst_batch[0])
                self.backward(pred, worst_batch[1])
        
        print('Finished Training at', total_iters, 'iterations. Final validation score:', score[-1])
        self.save(self.save_path)
        

    def split_dataset(self, spectra, labels, val_percentage=0.05):
        num_train = round((1-val_percentage)*spectra.shape[0])

        dataset_train = DataLoader(MLPDataset(spectra[:num_train], labels[:num_train]),
                                    batch_size=self.batch_size,
                                    num_workers=0,
                                    shuffle=True,
                                    drop_last=False)

        spectra_val = spectra[num_train:]
        labels_val = labels[num_train:]
        
        return dataset_train, (spectra_val, labels_val)

    def plot_loss(self, loss):
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        plt.plot(loss)
        plt.savefig(self.save_path+'loss.png', format='png')
        plt.cla()

    def plot_val_score(self, score: np.ndarray):
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        score=score.transpose()
        for met_score in score:
            plt.plot(met_score)
        plt.ylim([0,1])
        plt.savefig(self.save_path+'val_score.png', format='png')
        plt.cla()

    def save(self, path):
        torch.save(self.cpu().state_dict(), path+'.pth')
        print('Saved model at', path+'.pth')
        if self.gpu is not None:
            self.cuda()