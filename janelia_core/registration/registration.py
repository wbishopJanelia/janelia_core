""" Tools for registering imaging data.

    William Bishop
    bishopw@hhmi.org
"""

from dipy.align.transforms import TranslationTransform2D
from dipy.align.transforms import TranslationTransform3D
from dipy.align.imaffine import AffineRegistration
from dipy.align.imaffine import MutualInformationMetric
import h5py
import numpy as np
import pathlib
import pyspark
from scipy.ndimage import fourier_shift
from scipy.ndimage.filters import median_filter
from skimage.feature import register_translation
from time import time
from typing import Sequence
from typing import Iterable

from janelia_core.dataprocessing.dataset import DataSet
from janelia_core.dataprocessing.utils import get_image_data
from janelia_core.dataprocessing.image_stats import std_through_time
from janelia_core.math.basic_functions import l_th
from janelia_core.math.basic_functions import u_th
from janelia_core.math.basic_functions import divide_into_nearly_equal_parts
from janelia_core.fileio.exp_reader import read_img_file


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



def calc_phase_corr_shift(ref_img: np.ndarray, shifted_img: np.ndarray, **kwargs) -> np.ndarray:
    """ Calculates pixel-wise shift between two images using phase correlation.

    This function will return the shift to go from shifted_img to ref_img.

    This function works for both 2d and 3d images.

    Args:
        ref_img: The reference image as a numpy array.

        shifted_img: The shifted image as a numpy array.

        **kwargs: Extra keyword arguments to pass to the underlying registration call.

    Returns: The pixel-wise shift to go from the shifted_img to ref_img.
    """
    shift, _, _ = register_translation(ref_img, shifted_img, **kwargs)
    return shift


def apply_translation(base_img: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """ Applies a shift translation to an image.

    Translation will be performed in Fourier space.

    This function works for both 2d and 3d images.

    Args:
        base_img: The image to shift.

        shift: A vector of shifts in each dimension.

    Returns: The shifted image.
    """
    orig_dtype = base_img.dtype
    return np.ndarray.astype(np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base_img), shift))), orig_dtype)


def get_valid_translated_image_window(shifts: np.ndarray, image_shape: np.ndarray) -> tuple:
    """ Gets a window of an image which is still valid after a shift.

    Returns an window of a shifted image for which pixels in that window were
    shifted versions of pixels in an unshifted image.  This allows us to
    remove pixels from consideration in a shifted which were not based on
    pixels in the original image.

    Multiple shifts can be supplies.  In that case, the window will be valid for
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


def get_local_translations(ref_img: np.ndarray, shifted_img: np.ndarray, n_z: int, n_x: int, n_y: int,
                           sc: pyspark.SparkContext = None, **kwargs, ) -> list:
    """ Given two 3d images, calculates translations to align pieces of one to the other.

    This function will break up the image into roughly equally sized regions.
    It will then calculate the translation between the images in each region.

    The returned shifts will be for going from shifted_img to ref_img.

    Args:
        ref_img: The reference image to align to

        shifted_img: The shifted image

        n_z - The number of regions to create along the z-axis.

        n_x - The number of regions to create along the x-axis.

        n_y - The number of regions to create along the y-axis.

        sc: An optional Spark context to use to speed up computation.

        **kwargs: Optional keyword arguments to pass to the command to learn each local shift

    Returns:
            local_shifts: A list of shifts for each local region.  Shifts will be stored in a n_z by n_x by n_y by d array,
            where d is the dimension of the computed shifts.

            z_slices, x_slices, y_slices: List of slices for each region.  local_shifts[z, x, y, :] contains the shift
            for the region indexed by z_slices[z], x_slices[x] and y_slices[y].
    """

    # Determine indices of voxels in each region
    img_shape = ref_img.shape
    n_z_slices = divide_into_nearly_equal_parts(img_shape[0], n_z)
    n_x_slices = divide_into_nearly_equal_parts(img_shape[1], n_x)
    n_y_slices = divide_into_nearly_equal_parts(img_shape[2], n_y)

    start_z_inds = np.cumsum(np.concatenate((np.asarray([0]), n_z_slices)))
    start_x_inds = np.cumsum(np.concatenate((np.asarray([0]), n_x_slices)))
    start_y_inds = np.cumsum(np.concatenate((np.asarray([0]), n_y_slices)))

    z_slices = [slice(start_z_inds[i], start_z_inds[i] + n_z_slices[i], 1) for i in range(n_z)]
    x_slices = [slice(start_x_inds[i], start_x_inds[i] + n_x_slices[i], 1) for i in range(n_x)]
    y_slices = [slice(start_y_inds[i], start_y_inds[i] + n_y_slices[i], 1) for i in range(n_y)]

    slice_inds = list(np.ndindex(n_z, n_x, n_y))

    # Calculate the shift in each local region
    def get_local_shift(local_inds):
        print('Calculating local shift for region: '+ str(local_inds))
        z_slice = z_slices[local_inds[0]]
        x_slice = x_slices[local_inds[1]]
        y_slice = y_slices[local_inds[2]]
        return calc_phase_corr_shift(ref_img[z_slice, x_slice, y_slice], shifted_img[z_slice, x_slice, y_slice], **kwargs)

    if sc is None:
        local_shifts_list = []
        for slice_ind in slice_inds:
            local_shifts_list.append(get_local_shift(slice_ind))
    else:
        local_shifts_list = sc.parallelize(slice_inds).map(get_local_shift).collect()

    # Format output
    shift_dim = len(local_shifts_list[0])

    local_shifts = np.empty([n_z, n_x, n_y, shift_dim])
    for i, shift in enumerate(local_shifts_list):
        slice_ind = slice_inds[i]
        local_shifts[slice_ind[0], slice_ind[1], slice_ind[2], :] = shift

    return [local_shifts, z_slices, x_slices, y_slices]


def save_shifted_images(images: list, shifts: list, save_folder: pathlib.Path, sc: pyspark.SparkContext = None,
                        h5_dataset_name: str = 'data'):
    """ Applies a translational registration to each image in a list, saving the registered images.


        Images will be reduced in size to only include those voxels which are valid after registration for
        all time points.

        Files will be saved as h5 files.  This function currently expects shifts to specify translatoin in the x-y
        plane, and it will apply this same shift to all images.

    Args:
        images: A list of pathlib.Path objects providing the path to each file image to shift

        shifts: A list of shifts to apply to each image

        save_folder: The folder to save the shifted images in.

        sc: An optional SparkContext to use to speed up computation.

        h5_dataset_name: The name of the datasets for the shifted images in the saved h5 files

    """

    # Open the fist image to get image shape,
    image_1 = read_img_file(images[0])
    plane_shape = image_1.shape[1:3]

    valid_window = get_valid_translated_image_window(np.asarray(shifts), plane_shape)

    # Define a helper function to shift and save images
    def shift_and_save_image(im_ind):
        im_file = images[im_ind]
        print('Processing: ' + str(im_file))

        # Shift image
        raw_image = read_img_file(im_file)
        raw_image_type = raw_image.dtype
        for z in range(raw_image.shape[0]):
            raw_image[z, :, :] = np.ndarray.astype(apply_translation(raw_image[z, :, :], shifts[im_ind]),
                                                   raw_image_type)

        # Get the valid window for the image
        raw_image = raw_image[:, valid_window[0], valid_window[1]]

        # Save the image to file
        reg_im_file_name = save_folder / im_file.name
        with h5py.File(reg_im_file_name, 'w') as new_file:
            new_file.create_dataset(h5_dataset_name, raw_image.shape, raw_image.dtype, raw_image)

    # Process each file
    n_files = len(images)
    im_inds = np.asarray(range(n_files))
    if sc is None:
        for im_ind in im_inds:
            shift_and_save_image(im_ind)
    else:
        sc.parallelize(im_inds).foreach(shift_and_save_image)
