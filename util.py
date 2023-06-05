import pickle
import numpy as np
from scipy.interpolate import griddata

def save_result(result, filename):
    with open(filename, "wb") as f:
        pickle.dump(result, f)

def load_result(filename):
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result

def resample_irreg2regGrid(x,y,z,xi,yi, method='linear'):
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
    zi = griddata((x, y), z, (xi, yi), method=method)
    return xi, yi, zi