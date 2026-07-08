from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from argparse import Namespace

class TABDataset(Dataset):
    def __init__(self,file, split, window_size, step, data_ratio=1.0):
        super(TABDataset, self).__init__()
        self.file = file
        self.split = split
        self.window_size = window_size 
        self.step = step 
        self.data_ratio = data_ratio
        self.load_meta()
        self.load_data()
        if "swat" in file.lower():
            # self.down_sample(50)
            chs = [i for i in range(0, 51)]
            chs.remove(5)
            self.select_channel(chs) 
            # self.select_channel([0,1,2,3,4,5])
        elif "msl" in file.lower():
            self.select_channel([0])
        # elif "psm" in file.lower():
            # self.down_sample(20)
        
    
    def load_meta(self):
        meta_file = os.path.join("data","tab", "DETECT_META.csv")
        meta_df = pd.read_csv(meta_file)
        is_multivariate = meta_df.loc[meta_df["file_name"] == os.path.basename(self.file)]["if_univariate"].values[0] == False
        self.train_lens = meta_df.loc[meta_df["file_name"] == os.path.basename(self.file)]["train_lens"].values[0]
        self.file_dir = "multi_ts" if is_multivariate else "uni_ts"


    def load_data(self):
        data = read_data(os.path.join("data","tab", self.file_dir, os.path.basename(self.file)))
        scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(-1,1))
        if self.split == 'train':
            self.data = data.values[:self.train_lens,:-1]
            self.data = np.nan_to_num(self.data)
            self.label = np.zeros(self.data.shape[0])
            scaler.fit(self.data)
            self.data = scaler.transform(self.data)
            # Apply data ratio sampling
            if self.data_ratio < 1.0:
                num_samples = int(len(self.data) * self.data_ratio)
                self.data = self.data[:num_samples]
                self.label = self.label[:num_samples]
        elif self.split == 'test':
            train = data.values[:self.train_lens,:-1]
            train = np.nan_to_num(train)
            self.data = data.values[self.train_lens:,:-1]
            self.data = np.nan_to_num(self.data)
            self.label = data.values[self.train_lens:,-1:].reshape(-1)
            scaler.fit(train)
            self.data = scaler.transform(self.data)
        elif self.split == 'val':
            self.data = data.values[:self.train_lens,:-1]
            self.data = np.nan_to_num(self.data)
            self.label = np.zeros(self.data.shape[0])
            scaler.fit(self.data)
            self.data = scaler.transform(self.data)
        print(f"Loaded {self.split} data with shape {self.data.shape} and labels with shape {self.label.shape}")
    
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
    




import pandas as pd
from typing import Union, Any, Optional, Dict
import numpy as np

def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Read the data file and return DataFrame.If the data is spatial-temporal format,

    return it as a numpy array; otherwise, return it as a Pandas DataFrame.

    :param path: The path to the data file.
    :return:  The content of the data file.
    """
    data = pd.read_csv(path)
    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)
    
def is_st(data: pd.DataFrame) -> bool:
    """
    Checks if data of the CSV file are in spatial-temporal format.

    :param data: The series data.
    :return: Are all values in 'cols' column are in spatial-temporal format.
    """
    return data.shape[1] == 4 and "id" in data.columns


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    """
    Convert spatial-temporal data from a DataFrame

    to a three-dimensional(time stamp,feature,sensor)  numpy array.

    :param df: Spatial-temporal data.
    :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
    :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
    """
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

    sensors = df["id"].unique()
    features = df["cols"].unique()
    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )

    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
    data_np = np.transpose(data_np, (0, 2, 1))

    if nrows is not None:
        data_np = data_np[:nrows, :, :]

    return data_np


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.

    According to the provided file path, read the data file and return the corresponding DataFrame.

    :param data: Data frame to read.
    :return:  The DataFrame of the content of the data file.
    """
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df