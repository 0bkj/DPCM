from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from argparse import Namespace

class TSBDataset(Dataset):
    def __init__(self,file, split, window_size, step):
        super(TSBDataset, self).__init__()
        self.file = file
        self.split = split
        self.window_size = window_size 
        self.step = step 
        self.resolve_filename()
        self.load_data()
        if "msl" in self.file.lower():
            self.select_channel([0])
        if "swat" in self.file.lower():
            self.data = self.data[:,1:]
        print("data shape", self.data.shape)

    def resolve_filename(self):
        # 003_MSL_id_2_Sensor_tr_883_1st_1238.csv
        split_filename = self.file.split("_")
        self.train_len = int(split_filename[-3])

    def load_data(self):
        df = pd.read_csv(os.path.join("data","TSB-AD-M", self.file))
        data = df.values[:,:-1]
        label = df.values[:,-1:].reshape(-1)
        train = data[:self.train_len]
        test = data[self.train_len:]
        test_label = label[self.train_len:]
        scaler = StandardScaler()
        scaler.fit(train)
        if self.split == 'train':
            self.data = scaler.transform(train)
            self.label = np.zeros(train.shape[0])
        elif self.split == 'test':
            self.data = scaler.transform(test)
            self.label = test_label
        elif self.split == 'val':
            self.data = scaler.transform(train)
            self.label = np.zeros(train.shape[0])

    def __len__(self):
        return (self.data.shape[0] - self.window_size) // self.step + 1
    
    def __getitem__(self, index):
        index = index * self.step
        x = self.data[index: index + self.window_size]
        y = self.label[index: index + self.window_size]
        return np.float32(x), np.float32(y)
    
    def down_sample(self,step):
        self.data = self.data[::step]
        self.label = self.label[::step]
        print(f"down sample data with step {step} data shape {self.data.shape}")
    
    def select_channel(self,channel):
        self.data = self.data[:,channel]
        print(f"select channel data shape {self.data.shape}")

if __name__ == '__main__':
    dataset = TSBDataset("003_MSL_id_2_Sensor_tr_883_1st_1238.csv", "train", 100, 1)
    print(dataset.data.shape)
    print(dataset.data[:10])