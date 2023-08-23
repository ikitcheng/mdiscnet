from torch.utils.data import Dataset
import os
import numpy as np
import util
import pickle

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, labels_path: str, transform=None):
        self.labels = self.load_labels(labels_path)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """ Get a sample from the dataset based on the given index """
        data_path = os.path.join(self.root_dir,
                                self.labels[index])
        sample = self.load_data(data_path)
        label = self.labels[index][0:-4]
        
        if self.transform is not None:
            sample = self.transform(sample).float()
        return sample, label

    def load_data(self, data_path):
        data = util.load_result(data_path)
        return data
    
    def load_labels(self, labels_path):
        labels = np.array(util.load_txt(labels_path).strip().split('\n'))
        return labels

