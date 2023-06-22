import torch

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
        channel_sum += torch.sum(data)
        channel_sum_squared += torch.sum(data.pow(2))

        count += len(data) # number of points in each slice

    # Calculate the mean and standard deviation
    mean = channel_sum / count
    std = torch.sqrt((channel_sum_squared / count) - mean.pow(2))
    
    return mean, std