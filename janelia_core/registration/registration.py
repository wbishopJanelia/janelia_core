""" Tools for registering imaging data.

    William Bishop
    bishopw@hhmi.org
"""

import h5py
import numpy as np
import pathlib
import pyspark
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation

from janelia_core.math.basic_functions import l_th
from janelia_core.math.basic_functions import u_th
from janelia_core.math.basic_functions import divide_into_nearly_equal_parts
from janelia_core.fileio.exp_reader import read_img_file


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
