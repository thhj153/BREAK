from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        title = data[0]
        label = data[1]
        nodes = data[2]
        ids = data[3]

        return title, label, nodes, ids





