from torch.utils.data import DataLoader

def data_loaders(dataset, split, batch_size):
    if split == "train":
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=2)
    elif split == "test":
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)
    elif split == "val":
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)
    else:
        raise ValueError('Invalid split')
    return loader


def load_data_and_data_loaders(file, split, window_size, step, batch_size, data_ratio=1.0, sample_stride=None):
    if file.startswith("tsb"):
        from datasets.tsb_dataset import TSBDataset
        # remove the prefix "tsb_" from the file name
        file = file[4:]
        dataset = TSBDataset(file=file,split=split,window_size=window_size, step=step)
    elif file.endswith(".csv"):
        from datasets.tab_dataset import TABDataset
        dataset = TABDataset(
            file=file, split=split, window_size=window_size, step=step,
            data_ratio=data_ratio, sample_stride=sample_stride,
        )
    elif file.endswith(".npy"):
        from datasets.indices_dataset import IndicesDataset
        dataset = IndicesDataset(file=file,split=split,window_size=window_size, step=step, data_ratio=data_ratio)
    else:
        raise ValueError('Invalid file')
    
    loader = data_loaders(dataset, split, batch_size)

    return dataset, loader

