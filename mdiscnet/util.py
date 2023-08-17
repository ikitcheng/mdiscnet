import pickle
import json
import numpy as np
from scipy.interpolate import griddata
import torch

def save_model(model, model_path):
    """ Save the model weights. """
    torch.save(model.state_dict(), model_path)
    
def load_model(model, model_path):
    """ Load the model with saved weights on 
    the same device it was trained and saved on. """
    model.load_state_dict(torch.load(model_path))
    model.to(model.device)
    model.eval()
    return model

def save_result(result, filename):
    with open(filename, "wb") as f:
        pickle.dump(result, f)

def load_result(filename):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def resample_irreg2regGrid(x,y,z,xi,yi, interp_method='linear'):
    """ Resampling irregularly spaced data to a regular grid using griddata. """

    # Generate some irregularly spaced data
    x = x.flatten()  # x-coordinates
    y = y.flatten()  # y-coordinates
    z = z.flatten()  # data values at each (x, y) point

    # Define the regular grid
    # xi are the x-coordinates of the regular grid (could reduce the limits)
    # yi are the y-coordinates of the regular grid (could reduce the limits)
    xi, yi = np.meshgrid(xi, yi)  # create the meshgrid

    # Perform the interpolation
    zi = griddata((x, y), z, (xi, yi), method=interp_method)
    return xi, yi, zi