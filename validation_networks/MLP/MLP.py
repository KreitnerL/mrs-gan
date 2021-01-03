from models.auxiliaries.auxiliary import init_weights, set_num_dimensions
from models.networks import ExtractorMLP
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from torch.utils.data.dataloader import DataLoader
from validation_networks.MLP.MLP_dataset import MLPDataset

class MLP():
    def __init__(self, save_path:str, val_fun=None, gpu: int = None, in_out = (1,1), num_neurons = (100, 100, 100), lr=0.001, batch_size: int = 1, num_epoch: int = 15, validate_every: int = 2500):
        assert validate_every % batch_size == 0
        set_num_dimensions(1)
        self.network = ExtractorMLP(in_out, num_neurons, gpu_ids=[gpu], cbam=False)
        init_weights(self.network, "kaiming", activation='leaky_relu')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr, betas=(0.9, 0.999))
        self.loss_fn = nn.MSELoss()
        self.save_path = save_path
        self.batch_size = batch_size
        self.validate_every = validate_every
        self.num_epoch = num_epoch
        self.val_fun = val_fun
        self.gpu=gpu
        self.load(self.save_path+'.pth')

        
        if gpu is not None:
            torch.cuda.set_device(gpu)
            self.input = torch.cuda.FloatTensor(batch_size, 1, 1)
            self.label = torch.cuda.FloatTensor(batch_size, 1, 1)
            self.network.cuda()
        else:
            self.input = torch.Tensor(batch_size, 1, 1)
            self.label = torch.Tensor(batch_size, 1, 1)
            raise ValueError('MLP should run on a GPU!')

    def load(self, path):
        if os.path.isfile(path):
            self.network.load_state_dict(torch.load(path))
            self.pretrained = True
            print('Loaded pretrained model from', path)
        else:
            self.pretrained = False

    def backward(self, pred, label):
        loss = self.loss_fn(pred, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        dataset = DataLoader(MLPDataset(x, None), batch_size=self.batch_size,
                             num_workers=0, drop_last=False, shuffle=False)
        return self._predict(dataset)

    def _predict(self, val_dataset: DataLoader) -> np.ndarray:
        pred = []
        for spectra, _ in val_dataset:
            self.input.resize_(spectra.size()).copy_(spectra)
            pred.append((self.network.forward(self.input)*3.6).detach().cpu().numpy())

        return np.concatenate(pred)
        

    def train(self, spectra_train, labels_train, spectra_test, labels_test, tolerance: float = 1e-3, patience: int = 20):
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
        score = []
        early_stop = False
        early_stop_cond = lambda score: min(score[-patience:]) > min(score[:-patience])-tolerance 
        print('Training model...')
        while True:
            if early_stop:
                break
            for (spectra, labels) in dataset_train:
                total_iters += self.batch_size
                self.input.resize_(spectra.size()).copy_(spectra)
                self.label.resize_(labels.size()).copy_(labels)
                pred = self.network.forward(self.input)*3.6
                loss.append(self.backward(pred, self.label))

                if total_iters%self.validate_every==0:
                    self.plot_loss(loss, total_iters)

                    pred = self._predict(dataset_test)
                    score = np.append(score, self.val_fun(pred, labels_test))
                    self.plot_val_score(score, total_iters)

                    if len(score)>1 and score[-1]<min(score[:-1]):
                        self.save(self.save_path)
                    
                    if len(score)>patience and early_stop_cond(score):
                        early_stop=True
                        break
        
        self.load(self.save_path+'.pth')
        print('Finished Training at', total_iters, 'iterations. Final validation score:', min(score))

    def plot_loss(self, loss, total_iters):
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        x = np.linspace(total_iters / len(loss), total_iters, len(loss))
        plt.plot(x, loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig(self.save_path+'loss.png', format='png')
        plt.cla()

    def plot_val_score(self, score: np.ndarray, total_iters):
        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
        x = np.linspace(total_iters / len(score), total_iters, len(score))
        plt.plot(x, score)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Reltative Error')
        plt.savefig(self.save_path+'val_score.png', format='png')
        plt.cla()

    def save(self, path):
        torch.save(self.network.cpu().state_dict(), path+'.pth')
        print('Saved model at', path+'.pth')
        if self.gpu is not None:
            self.network.cuda()