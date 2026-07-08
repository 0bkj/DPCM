from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from argparse import Namespace


class IndicesDataset(Dataset):
    def __init__(self,file, split, window_size, step, data_ratio=1.0):
        super().__init__()
        self.file = file
        self.split = split
        self.window_size = window_size 
        self.step = step
        self.data_ratio = data_ratio
        self.load_data()
       
    def load_data(self):
        # scaler = StandardScaler()
        data = np.load(os.path.join("data","indices",self.file))
        if self.split == 'train':
            self.data = data
            # Apply data ratio sampling
            if self.data_ratio < 1.0:
                num_samples = int(len(self.data) * self.data_ratio)
                self.data = self.data[:num_samples]
        elif self.split == 'test':
            self.data = data
        elif self.split == 'val':
            self.data = data
        print(f"Loaded {self.split} data with shape {self.data.shape}")
    
    def __len__(self):
        return (self.data.shape[0] - self.window_size) // self.step + 1
    
    def __getitem__(self, index):
        index = index * self.step
        x = self.data[index: index + self.window_size]
        return np.float32(x)
    