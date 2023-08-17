import torch
import numpy as np

def online_mean_std(custom_dataset):
    """ Online estimation of mean and standard deviation of single channel grid data. """
    # Accumulate the sum and squared differences
    channel_sum = 0
    channel_sum_squared = 0
    count = 0

    for data in custom_dataset:
        # Flatten the image tensor
        data = data[0].flatten() # data[1] is the label

        # Accumulate the sum and squared differences
        channel_sum += torch.nansum(data)
        channel_sum_squared += torch.nansum(data.pow(2))

        count += len(data) # number of points in each slice

    # Calculate the mean and standard deviation
    mean = channel_sum / count
    std = torch.sqrt((channel_sum_squared / count) - mean.pow(2))
    
    return mean, std

class ClipTransform:
    """ Define a custom clip transformation"""
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data):
        return torch.clip(data, self.min_value, self.max_value)
    
class ReplaceNaN:
    """ Define a custom transformation to replace NaN values with a constant """
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, data):
        data = np.array(data)
        data[np.isnan(data)] = self.constant
        return torch.from_numpy(data)