""" Tools for efficiently calculating statistics over images.

    This module is geared specifically towards working with time series of volumetric data.

    William Bishop
    bishopw@hhmi.org
"""

from operator import add

import numpy as np
import pyspark

from janelia_core.dataprocessing.utils import get_image_data


SPARK_N_IMGS = 100 # Number of images in a calculation we must exceed to use spark


def std_through_time(images: list, sc: pyspark.SparkContext=None, verbose=True,
                     correct_denom=True, h5_data_group: str='default') -> dict:
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

        h5_data_group: The hdfs data group holding image data in hdfs files

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
        moments = sc.parallelize(images).map(
            lambda im: calc_uc_moments(get_image_data(im, h5_data_group=h5_data_group))).reduce(list_sum)
    else:
        if verbose:
            print('Processing ' + str(n_images) + ' images without spark.')
            moments = calc_uc_moments(get_image_data(images[0], h5_data_group=h5_data_group))
            for i in images[1:]:
                moments = list_sum(moments, calc_uc_moments(get_image_data(i, h5_data_group=h5_data_group)))

    var = moments[1] - np.square(moments[0])
    var[np.where(var < 0)] = 0  # Correct for small floating point errors
    std = np.sqrt(var)

    if correct_denom:
        std = (n_images/(n_images - 1))*std

    if verbose:
        print('Done processing ' + str(n_images) + ' images.')

    return {'std': std, 'mean': moments[0], 'sec_mom': moments[1]}


def create_brain_mask(stats: dict, std_p: int=70, mean_p: int=70, verbose=True) -> np.ndarray:
    """ Creates a brain mask.

    Given a structure of statistics produced by std_through_time, this function returns a binary array estimating which
    voxels belong to the brain.

    The array is estimated by looking for voxels with both a mean and standard deviation which surpass a user defined
    threshold.

    Args:
        stats: The dictionary of statistics produced by std_through_time.

        std_p: The threshold percentile for standard deviation values

        mean_p: The threshold percentile for mean values

        verbose: True if status updates should be printed.

    Returns:
         mask: A binary array the same shape as an image.  Each entry indicates if the corresponding voxel in the images
         belongs to the brain or not.

    """

    std_t = np.percentile(stats['std'], std_p)
    if verbose:
        print(str(std_p) + '-th standard deviation percentile: ' + str(std_t))
    mean_t = np.percentile(stats['mean'], mean_p)
    if verbose:
        print(str(mean_p) + '-th mean percentile: ' + str(mean_t))

    return np.bitwise_and(stats['std'] > std_t, stats['mean'] > mean_t)


def identify_rois_in_brain_mask(brain_mask: np.ndarray, rois: list, p_in=.9) -> np.ndarray:
    """
    Identifies ROIs that significantly overlap with a brain mask.

    Args:
        brain_mask: A binary np.ndarray the same shape as the shape of images rois were extracted from, indicating
        which voxels belong to the brain.

        rois: A list of rois.  Each entry should have a dictionary with entries 'x', 'y' and 'z' listing the x, y and z
        coordinates of an roi.

        p_in: The percentage of voxels of an roi that must overlap with the brain mask to be considered in the mask.

    Returns:
        rois_in_brain:
    """
    rois_in_brain = np.where([np.sum(brain_mask[roi['z'], roi['y'], roi['x']])/len(roi['z']) > p_in for roi in rois])
    return rois_in_brain[0]
