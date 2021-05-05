import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize():
        pass

    def __getattr__(self, item):
        if item == 'magnitude':
            mag = self.magnitude()
            return mag
