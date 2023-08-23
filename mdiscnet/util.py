import pickle
import json
import numpy as np
from scipy.interpolate import griddata
import torch
import ast


def save_model(model, model_path):
    """ Save the model weights. """
    torch.save(model.state_dict(), model_path)
    
def load_model(model, model_path, device):
    """ Load the model with saved weights on 
    the same device it was trained and saved on. """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(model.device)
    model.eval()
    return model

def save_result(result, filename):
    """ Save result to .pkl file """
    with open(filename, "wb") as f:
        pickle.dump(result, f)

def load_result(filename):
    """ Load result from .pkl file """
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result

def save_json(data, filename):
    """ Save data to .json file """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_txt(data: list, filename: str):
    """ save a list or array to .txt file """
    if isinstance(data, np.ndarray) :
        data = data.tolist()
        
    with open(filename, "w") as f:
        f.write('\n'.join(data))

def load_txt(filename: str, literal_eval=False):
    """ Load data from .txt file (optional literal eval) """
    with open(filename, 'r') as f:
        data = f.read()
    
    if literal_eval:
        data = ast.literal_eval(data)
    return data

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