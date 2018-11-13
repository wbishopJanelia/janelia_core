""" Utilities for working with imaging data:

    William Bishop
    bishopw@hhmi.org
"""

import types

import numpy as np
import pyspark


from janelia_core.fileio.exp_reader import read_img_file


def get_image_data(image) -> np.ndarray:
    """ Gets image data for a single image.

    This is a wrapper that allows us to get image data for a single image from a file or from a numpy array
    seamlessly in our code.  If image is already a numpy array, image is simply returned
    as is.  Otherwise, image is assumed to be a path to a image which is opened and the data
    is loaded and returned as a numpy array.

    Args:
        image: Either a numpy array or path to the image.

    Returns: The image data.
    """
    if isinstance(image, np.ndarray):
        return image
    else:
        return read_img_file(image)


def get_processed_image_data(images: list, func: types.FunctionType = None, sc: pyspark.SparkContext = None) -> list:
    """ Gets processed image data for multiple images.
    
    This is a wrapper that allows retrieving images from files or from numpy arrays,
    applying light processing independently to each image and returning the result.  
    
    Args:
        images: A list of images.  Each entry is either a numpy array or a path to an image file.
        
        func: A function to apply to each image.  If none, images will be returned unaltered.
        
        sc: An optional pySpark.SparkContext object to use in speeding up reading of images.
        
    Returns: The processed image data as a list.  Each processed image is an entry in the list. 
    """

    if sc is None:
        return [func(get_image_data(img)) for img in images]
    else:
        def _process_img(img):
            return func(get_image_data(img))

        return sc.parallelize(images).map(_process_img).collect()
