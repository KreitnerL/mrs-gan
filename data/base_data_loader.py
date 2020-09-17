
class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data(self):
        return #self#.dataloader

    def save_path(self):
        self.save_filename = 'model_phase_indices.csv'
        #return self.save_filename

    def splitData(dataset, val_split=0.2, test_split=0.1):
        return {}
        
