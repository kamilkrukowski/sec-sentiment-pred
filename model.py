import torch.nn as nn
from torch.utils.data import Dataset
import torch

class Dataset_8K(Dataset):

    def __init__(self, split='train'):

        self.x = []
        self.y = []

        if split == 'train':
            fpath = 'train.csv'
            self._load_file_(fpath);

        if split == 'test':
            fpath = 'test.csv'
            self._load_file_(fpath);

    def _load_file_(self, fpath):
        with open(fpath, 'r') as f:
            _ = f.readline() # Header to delete
            for line in f.readlines():
                curr_x, curr_y, *r = tuple(line.split(','))
                self.x.append(curr_x)
                self.y.append(int(curr_y))

    def __len__(self):
        return len(self.y)
  
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

data = Dataset_8K()