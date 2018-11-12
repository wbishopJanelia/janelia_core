""" Utilities for working with imaging data:

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np

from janelia_core.fileio.exp_reader import read_img_file


def get_image_data(image) -> np.ndarray:
    """ Gets image data.

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
