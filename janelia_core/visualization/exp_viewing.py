""" Tools for visualizing imaging experiment data.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np

import janelia_core.dataprocessing.dataset

def visualize_exp(dataset: janelia_core.dataprocessing.dataset.DataSet,
                  cont_var_key: str, image_keys: list):
    """ Function for visually exploring the data in a Dataset object.

    This function will create a GUI that plots continuous variables through time, allowing
    the user to zoom in on the continuous variables.  After zooming in on a region of the
    continuous variables, the user can load the images for that region.

    Args:
        dataset: The dataset to explore.

        cont_var_key: The key of the dictionary containing continuous variables to examine.

        image_keys: A list of keys to dictionaries containing image series to examine.
    """
    pass

def view_images_with_continuous_values(cont_var_dict: dict, image_dicts: list, cont_var_labels: list = None,
                                       image_labels: list = None)
    """ Function to explore continuous variables and series of images together.
    
    Args: 
        cont_var_dict: A dictionary containing data for a time series of continuous values.  Must have two keys.  
        The first, 'ts',is a 1-d numpy.ndarray with timestamps.  The second 'vls' is a list of or numpy array of data
        for each point in ts, where each row of data is a vector of continuous values for a time point.
        
        image_dics: A list of dictionaries. Each dictionary contains data for one time series of images. These
        dictionaries must have a 'ts' key, with timestamps (as cont_var_dict) and a 'vls' key, which will contain
        either (1) a 3-d numpy.ndarray of dimensions n_ts*n_pixels_n_pixels or (2) a list of paths to image files of
        length n_ts. 
        
        
    
    
    
    """