import torch
from sklearn.model_selection import StratifiedKFold

class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass


    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified batch sampling"""

    def __init__(self, y, batch_size, random_state, shuffle=True):

        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'

        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.rnd_state = random_state
        self.n_batches = int(len(y) / batch_size)


    def __iter__(self):
        skf = StratifiedKFold(n_splits = self.n_batches,
                              shuffle = self.shuffle,
                              random_state = self.rnd_state)
        for train_idx, test_idx in skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':

    from torch.utils.data import DataLoader, TensorDataset
     
    # Creating a toy dataset
    y1 = torch.zeros([50,1])
    y2 = torch.ones([12, 1])
    y = torch.cat([y1, y2], axis = 0).squeeze()
    x = torch.ones([len(y), 2])
    x[10,:] = 3.14
    x[-21,:] = 392
    
    # Using StratiFiedSampler
    dataset = TensorDataset(x, y)
    my_sampler = StratifiedSampler(y,10,10)
    loader = DataLoader(dataset, batch_sampler = my_sampler)
    
    # Checking results
    for x, y in loader:
        print(x)




