""" Tools for efficiently calculating statistics over images.

    This module is geared specifically towards working with time series of volumetric data.

    William Bishop
    bishopw@hhmi.org
"""

from operator import add
import time
from typing import Callable

from morphsnakes import morphological_chan_vese
import numpy as np
import pyspark

from janelia_core.dataprocessing.utils import get_reg_image_data


def std_through_time(images: list, image_slice: slice = slice(None, None, None),
                     t_dict: dict = None, sc: pyspark.SparkContext=None,
                     preprocess_func:Callable[[np.ndarray], np.ndarray] = None,
                     verbose=True, correct_denom=True, h5_data_group: str='default') -> dict:
    """ Calculates standard deviation of individual voxels through time.

    This function will also return the uncentered first (i.e, the mean) and second moments for each individual
    voxel through time as well, since these are calculated in the intermediate computations.

    Args:
        images: A list of either (1) numpy arrays containing the images or (2) file paths to the image files

        image_slice: The slice of each image to extract.  If registration is being performed, (see t_dict),
        coordinates of image_slice should be for the images after registration.

        t_dict: If this is not none, images will be registered before any statistics are calculated.  This dictionary
        has two entries:
            transforms: A list of registration transforms to apply to the images as they are being read in.  If none,
            no registration will be applied.

            image_shape: This is the shape of the original images being read in.

        sc: If provided, spark will be used to distribute computation.

        preprocess_func: A function to apply to the data of each image before calculating means.  If none,
        a function which returns image data unchanged will be used.

        verbose: True if progress updates should be printed to screen.

        correct_denom: If true, standard deviation is calculated by dividing by sqrt(n - 1) where n is the
        number of images.  If false, division by sqrt(n) is used.

        h5_data_group: The hdfs data group holding image data in hdfs files

    Returns:
        A dict with the standard deviation, mean and uncentered second moments for each voxel as numpy arrays. The keys
        will be 'std', 'mean' and 'sec_mom', where 'sec_mom' is the uncentered second moment.
    """

    n_images = len(images)
    do_reg = t_dict is not None

    # Define helper functions
    if preprocess_func is None:
        def preprocess_func(x):
            return x

    def calc_uc_moments(data: np.ndarray) -> list:
        return [(1/n_images)*data.astype('float32'), (1/n_images)*np.square(data.astype('float32'))]

    def list_sum(list_1: list, list_2: list) -> list:
        return list(map(add, list_1, list_2))

    # Package images together with transformss
    if do_reg:
        image_transforms = t_dict['transforms']
        image_shape = t_dict['image_shape']
    else:
        image_transforms = [None]*n_images
        image_shape = None
    image_data_list = list(zip(images, image_transforms))

    def get_im(im_data):
        im = im_data[0]
        t = im_data[1]
        return get_reg_image_data(image=im, image_shape=image_shape, t=t,
                                  image_slice=image_slice, h5_data_group=h5_data_group)

    # Perform calculations
    if sc is not None:
        print('Processing ' + str(n_images) + ' images with spark.')
        moments = sc.parallelize(image_data_list).map(
            lambda im: calc_uc_moments(preprocess_func(get_im(im)))).reduce(list_sum)
    else:
        if verbose:
            print('Processing ' + str(n_images) + ' images without spark.')
            moments = calc_uc_moments(preprocess_func(get_im(image_data_list[0])))
            for i_tuple in image_data_list[1:]:
                moments = list_sum(moments, calc_uc_moments(preprocess_func(get_im(i_tuple))))

    var = moments[1] - np.square(moments[0])
    var[np.where(var < 0)] = 0  # Correct for small floating point errors
    std = np.sqrt(var)

    if correct_denom:
        std = (n_images/(n_images - 1))*std

    if verbose:
        print('Done processing ' + str(n_images) + ' images.')

    return {'std': std, 'mean': moments[0], 'sec_mom': moments[1]}


def create_morphsnakes_brain_mask(img: np.ndarray, p: np.ndarray, morph_params: dict = {}, verbose=True)-> np.ndarray:
    """ Creates a brain mask using the morphsnakes package.

    This function operates independently on each plane.

    Args:
        img: The image to find the brain mask of.

        p: The percentile threshold value to use when forming the initial brain mask for each plane.  Specifically,
        for each plane in the image, the p-th percentile is calculated.  The initial mask is then formed by masking out
        any value less than this. p[i] contains the percentile for plane i.

        morphsnake_params: A dictionary of parameters to pass into morphological_chan_vese.  See that function
        for more information. Reasonable initial values are:
            morph_params = {'iterations': 50, 'smoothing': 4, 'lambda1': 1, 'lambda2': 1}

        verbose: True if progress updates should be printed.

    Returns:
        mask: A binary array the same shape as an image.  Each entry indicates if the corresponding voxel in the images
         belongs to the brain or not.

    """

    t0 = time.time()

    flat_image = False
    if img.ndim < 3:
        flat_image = True
        img = np.expand_dims(img, axis=0)

    # Define helper function here
    def process_plane(plane_img, p_i):
        th_vl = np.quantile(plane_img.reshape(plane_img.size), p_i)
        init_mask = plane_img <= th_vl
        return morphological_chan_vese(plane_img, init_level_set=init_mask, **morph_params)

    n_planes = img.shape[0]
    mask = np.zeros_like(img, dtype=np.uint8)
    for z_i in range(n_planes):
        mask[z_i, :, :] = process_plane(np.squeeze(img[z_i, :, :]), p[z_i])
        if verbose:
            print('Done extracting mask for plane ' + str(z_i + 1) + ' of ' + str(n_planes) + '.' +
                  ' Elapsed Time: ' + str(time.time() - t0))

    if flat_image:
        mask = np.squeeze(mask)

    return np.logical_not(mask)


def create_threshold_brain_mask(stats: dict, std_p: int=70, mean_p: int=70, verbose=True) -> np.ndarray:
    """ Creates a brain mask by thresholding statistics.

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
        rois_in_brain: A np.ndarray of indices of ROIs that are in the brain mask
    """
    rois_in_brain = np.where([np.sum(brain_mask[roi.voxel_inds])/roi.n_voxels() >= p_in for roi in rois])
    return rois_in_brain[0]


def identify_small_rois(roi_extents: np.ndarray, rois: list):
    """ Identifies ROIS that fit within a rectangular box of a set size.

    Args:
        roi_extents: A 3d array of the lengths of the sides of a rectangle (in pixels)
        rois must fit into.

        rois: A list of roi objects.

    Returns:
        small_rois: A np.ndarray of indices of rois that fit within the rectangle
    """

    small_rois = list()
    for i, roi in enumerate(rois):
        roi_bounding_box = roi.extents()
        if np.all(np.less_equal(roi_bounding_box, roi_extents)):
            small_rois.append(i)
    return np.asarray(small_rois)
