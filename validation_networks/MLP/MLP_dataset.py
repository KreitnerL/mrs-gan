from torch.utils.data import Dataset
import numpy as np
import torch
T= torch.Tensor

class MLPDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data)
        if labels is not None:
            self.labels = torch.from_numpy(labels)
        else:
            self.labels=None

    def __getitem__(self, index):
        return self.data[index], self.labels[index] if self.labels is not None else 0

    def __len__(self):
        return len(self.data)