""" Tools for registering imaging data.

    William Bishop
    bishopw@hhmi.org
"""

import dipy
from dipy.align.transforms import TranslationTransform2D
from dipy.align.transforms import TranslationTransform3D
from dipy.align.imaffine import AffineRegistration
from dipy.align.imaffine import MutualInformationMetric
import numpy as np
import pyspark
from scipy.ndimage.filters import median_filter

from time import time
from typing import Sequence
from typing import Iterable

from janelia_core.dataprocessing.dataset import DataSet
from janelia_core.dataprocessing.utils import get_image_data
from janelia_core.dataprocessing.image_stats import std_through_time
from janelia_core.math.basic_functions import l_th
from janelia_core.math.basic_functions import u_th


def calc_dataset_ref_image(dataset: DataSet, img_field: str, ref_inds: np.ndarray,
                           median_filter_shape: Sequence[int] = [1, 5, 5],
                           sc: pyspark.SparkContext = None) -> np.ndarray:
    """ Calculates a reference image.

        This function will:
            1) Median filter each plane in all images.
            2) Calculate a reference image by taking the mean of the mediate filtered images.

        Args:
            dataset: The dataset containing the images to register.

            img_field: The entry in dataset.ts_data containing the images.

            ref_inds: The indices of images to use for calculating the reference image.

            median_filter_shape: The shape of the median filter to use.

            sc: A spark context to use to distribute computations.  If none, no spark context will be used.


        Returns:
            ref_image: The reference image.

        """
    t0 = time()

    def med_filter_image(img):
        return median_filter(img.astype('float32'), median_filter_shape)

    print('Calculating reference image for registration.')
    ref_image_files = [dataset.ts_data[img_field]['vls'][i]['file'] for i in ref_inds]
    ref_stats = std_through_time(ref_image_files, preprocess_func=med_filter_image, sc=sc)
    t1 = time()
    print('Done calculating reference image.  Elapsed time: ' + str(t1 - t0))

    return ref_stats['mean']


def register_dataset_images(dataset: DataSet, img_field: str, ref_image: np.ndarray,
                            median_filter_shape: Sequence[int] = [1, 5, 5], reg_params: dict() = None,
                            t_field_name: str = 'shift_transform', reg_inds: np.ndarray = None, over_write:bool =True,
                            sc: pyspark.SparkContext = None,):

    """ Calculates transforms to register images through time.

    Currently this function calculates translations in x-y.  This function will:
        1) Median filter each plane in all images.
        2) Calculate the optimal translation to align the max projection of each median filtered image to the
           the reference image.

    Args:
        dataset: The dataset containing the images to register.

        img_field: The entry in dataset.ts_data containing the images.

        ref_image: The reference image to align to.

        median_filter_shape: The shape of the median filter to use.

        reg_params: A dictionary of optional parameters to provide to the call to estimate_translation

        t_field_name: Tha name to save the transform under with each image entry in the dataset.

        reg_inds: Indices of images to register.  If None, all images will be registered (unless over_write is set
        to False, in which case only unregistered images will be registered.)

        over_write: If true, all images are registered.  If false, only those that do not yet have a
        transform entry in their dictionary are registered.

        sc: A spark context to use to distribute computations.  If none, no spark context will be used.

    Returns: None. Each entry in the images list (which is a dictionary) will have an the transform
    for that image added as a 'transform' entry of the dictionary.

    """

    t0 = time()

    if reg_params is None:
        reg_params = dict()

    def med_filter_image(img):
        return median_filter(img.astype('float32'), median_filter_shape)

    # Calculate transforms for each image
    def reg_func(img):
        img = get_image_data(img)
        img = med_filter_image(img)
        img = np.max(img, 0)
        return estimate_translation(ref_image, img, **reg_params)

    # Determine which images we still need to register
    img_dicts = dataset.ts_data[img_field]['vls']
    n_images = len(img_dicts)

    if reg_inds is None:
        reg_inds = np.arange(0, n_images, 1)

    # See what images have not been registered
    if not over_write:
        unreg_image_inds = np.where([t_field_name not in d for d in img_dicts])[0]
    else:
        unreg_image_inds = np.arange(0, n_images)

    # Only register those images that we are marked as not registered and user has specified we are to register
    unreg_image_inds = np.intersect1d(unreg_image_inds, reg_inds)

    # Register images
    unreg_images = [img_dicts[i]['file'] for i in unreg_image_inds]
    n_unreg_images = len(unreg_images)

    print('Calculating registration transforms.')
    if sc is None:
        print('Processing ' + str(n_unreg_images) + ' images without spark.')
        transforms = [reg_func(img) for img in unreg_images]
    else:
        print('Processing ' + str(n_unreg_images) + ' images with spark.')
        transforms = sc.parallelize(unreg_images).map(reg_func).collect()
    t1 = time()
    print('Done calculating registration transforms.  Elapsed time: ' + str(t1 - t0))

    # Put transforms with images
    for t_i, t in enumerate(transforms):
        img_dicts[unreg_image_inds[t_i]][t_field_name] = t


def estimate_translation(fixed: np.ndarray, moving: np.ndarray, metric_sampling: float=1.0,
                         factors: Iterable =(4, 2, 1), level_iters: Iterable =(1000, 1000, 1000),
                         sigmas: Iterable =(8, 4, 1)):
    """

    Estimate translation between 2D or 3D images using dipy.align.

    This is code provided by Davis Bennett with minor reformatting by Will Bishop.

    Args:

    fixed : numpy array, 2D or 3D.  The reference image.

    moving : numpy array, 2D or 3D.  The image to be transformed.

    metric_sampling : float, within the interval (0,  1]. Fraction of the metric sampling to use for optimization

    factors : iterable.  The image pyramid factors to use

    level_iters : iterable. Number of iterations per pyramid level

    sigmas : iterable. Standard deviation of gaussian blurring for each pyramid level
    """

    metric = MutualInformationMetric(32, metric_sampling)

    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors, verbosity=0)

    if fixed.ndim == 2:
        transform = TranslationTransform2D()
    elif fixed.ndim == 3:
        transform = TranslationTransform3D()
    tx = affreg.optimize(fixed, moving, transform, params0=None)

    return tx


def apply_2d_dipy_affine_transform(moving_imgs: np.ndarray, t: dipy.align.imaffine.AffineMap) -> np.ndarray:
    """ Applies a 2d affine translation to a stack of images.

    This function works for a 2d image and a 3d stack of images.

    Args:
        moving_imgs: The image to shift. If a stack it should be of shape n_imgs*img_dim_1*img_dim_2.

        t: The transform to apply.

    Returns: The shifted image.
    """

    # If we have a single image, still put it in a trivial stack to make processing below standard
    is_stack = moving_imgs.ndim == 3
    if not is_stack:
        moving_imgs = np.expand_dims(moving_imgs, 0)

    # Get basic dimensions of stack
    stack_shape = moving_imgs.shape
    n_imgs = stack_shape[0]
    img_dims = stack_shape[1:3]

    # Setup transform to work with the shape of the images we are dealing with
    t.domain_shape = img_dims
    t.codomain_shape = img_dims

    # Apply shifts
    shifted_imgs = np.empty(stack_shape)
    for img_i in range(n_imgs):
        shifted_imgs[img_i, :, :] = t.transform(moving_imgs[img_i, :, :])

    # If user provided a single image, get rid of the extra stack dimension
    if not is_stack:
        shifted_imgs = np.squeeze(shifted_imgs)

    return shifted_imgs


def get_valid_translated_image_window(shifts: np.ndarray, image_shape: np.ndarray) -> tuple:
    """ Gets a window of an image which is still valid after a shift.

    Returns an window of a shifted image for which pixels in that window were
    shifted versions of pixels in an unshifted image.  This allows us to
    remove pixels from consideration in a shifted which were not based on
    pixels in the original image.

    Multiple shifts can be supplied.  In that case, the window will be valid for
    all shifts.  (Useful when finding valid windows for time series of shifted images).

    Args:
        shifts: The shifts to calculate the window for.  Each row is a shift.

        image_shape: The shape of the image.  Dimensions should be listed here in the
        same order they are listed in shifts.

    Returns: A tuple.  Each entry contains valid indices for a dimension, so that the valid window for an image
    can be recovered as image[t], if t is the returned tuple.

    """
    if shifts.ndim == 1:
        shifts = np.reshape(shifts, [shifts.size, 1])
        shifts = shifts.T

    shift_margins = np.sign(shifts) * np.ceil(np.abs(shifts))
    shift_maxs = np.ndarray.astype(np.max(shift_margins, 0), np.int)
    shift_mins = np.ndarray.astype(np.min(shift_margins, 0), np.int)

    shift_ups = l_th(shift_maxs, 0)
    shift_downs = u_th(shift_mins, 0)

    n_dims = shifts.shape[1]
    return tuple(slice(shift_ups[i], image_shape[i] + shift_downs[i], 1) for i in range(n_dims))
