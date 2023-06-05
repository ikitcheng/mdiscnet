from torch.utils.data import Dataset
import pickle

class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data, self.labels = self.load_data(data_path, labels_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get a sample from the dataset based on the given index
        sample = self.data[index]
        label = self.labels[index]
        
        # Apply the transforms, if provided
        if self.transform is not None:
            sample = self.transform(sample).float()

        return sample, label

    def load_data(self, data_path, labels_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        return data, labels
