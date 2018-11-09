""" Tools for efficiently calculating statistics over images.

    This module is geared specifically towards working with time series of volumetric data.

    William Bishop
    bishopw@hhmi.org
"""

from operator import add

import numpy as np
import pyspark

from janelia_core.fileio.exp_reader import read_img_file


SPARK_N_IMGS = 100 # Number of images in a calculation we must exceed to use spark


def get_image_data(image) -> np.ndarray:
    """ Gets image data.

    This is a wrapper that allows us to get image data from file or from a numpy array
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


def std_through_time(images: list, sc: pyspark.SparkContext = None, verbose = True, correct_denom = True) -> dict:
    """ Calculates standard deviation of individual voxels through time.

    This function will also return the uncentered first (i.e, the mean) and second moments for each individual
    voxel through time as well, since these are calculated in the intermediate computations.

    Args:
        images: A list of either (1) numpy arrays containing the images or (2) file paths to the image files

        sc: If provided, and there are more than SPARK_N_IMAGES images, spark will be used to
        distribute computation.

        verbose: True if progress updates should be printed to screen.

        correct_denom: If true, standard deviation is calculated by dividing by sqrt(n - 1) where n is the
        number of images.  If false, division by sqrt(n) is used.

    Returns:
        A dict with the standard deviation, mean and uncentered second moments for each voxel as numpy arrays. The keys
        will be 'std', 'mean' and 'sec_mom', where 'sec_mom' is the uncentered second moment.
    """

    n_images = len(images)

    # Define helper functions
    def calc_uc_moments(data: np.ndarray) -> list:
        return [(1/n_images)*data, (1/n_images)*np.square(data.astype('uint64'))]

    def list_sum(list_1: list, list_2: list) -> list:
        return list(map(add, list_1, list_2))

    # Perform calculations
    if sc is not None and n_images > SPARK_N_IMGS:
        print('Processing ' + str(n_images) + ' images with spark.')
        moments = sc.parallelize(images).map(lambda im: calc_uc_moments(get_image_data(im))).reduce(list_sum)
    else:
        if verbose:
            print('Processing ' + str(n_images) + ' images without spark.')
            moments = calc_uc_moments(get_image_data(images[0]))
            for i in images[1:]:
                moments = list_sum(moments, calc_uc_moments(get_image_data(i)))

    var = moments[1] - np.square(moments[0])
    var[np.where(var < 0)] = 0  # Correct for small floating point errors
    std = np.sqrt(var)

    if correct_denom:
        std = (n_images/(n_images - 1))*std

    if verbose:
        print('Done processing ' + str(n_images) + ' images.')

    return {'sqrt': std, 'mean': moments[0], 'sec_mom': moments[1]}

