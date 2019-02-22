""" Utilities for working with imaging data:

    William Bishop
    bishopw@hhmi.org
"""

import pathlib
import types

import dipy.align.imaffine
import h5py
import numpy as np
import os
import pyspark

from janelia_core.fileio.exp_reader import read_img_file
from janelia_core.math.basic_functions import is_standard_slice
from janelia_core.math.basic_functions import slice_contains


def get_image_data(image, img_slice: slice = slice(None, None, None), h5_data_group: str = 'default') -> np.ndarray:
    """ Gets image data for a single image.

    This is a wrapper that allows us to get image data for a single image from a file or from a numpy array
    seamlessly in our code.  If image is already a numpy array, image is simply returned
    as is.  Otherwise, image is assumed to be a path to a image which is opened and the data
    is loaded and returned as a numpy array.

    Args:
        image: Either a numpy array or path to the image.

        img_slice: The slice of the image that should be returned

        h5_data_group: The hdfs group holding image data in h5 files.

    Returns: The image data.
    """
    if isinstance(image, np.ndarray):
        return image[img_slice]
    else:
        return read_img_file(pathlib.Path(image), img_slice=img_slice, h5_data_group=h5_data_group)


def get_reg_image_data(image, image_slice: slice, image_shape: np.ndarray = None,
                       t: dipy.align.imaffine.AffineMap = None, h5_data_group: str = 'default'):
    """ Gets registered image data for a single image.

    This is a wrapper around get_image_data that allows the user to register an image before getting data from it.

    Args:
        image: Either a numpy array or path to the image.

        image_shape: The shape of the original image.  This is used to check that the requested window contains
        valid data after registration.  If t is none, this argument does not need to be supplied.

        t: The registration transform.  If this is none, no registration will be performed.

        image_slice: The slice of the image that should be returned.  Coordinates are *after* image registration.

        h5_data_group: The hdfs group holding image data in h5 files.

    Returns: The image data

    Raises:
        ValueError: If the requested slice for the registered image includes voxels for which there was no data in the
        original image to calculate the registered image for.
    """

    # Add imports from registration here to avoid circular imports
    from janelia_core.registration.registration import apply_2d_dipy_affine_transform
    from janelia_core.registration.registration import get_valid_translated_image_window

    # Just call get_image_data if we are not doing registration.
    if t is None:
        return get_image_data(image, img_slice=image_slice, h5_data_group=h5_data_group)

    # Helper function to form slices we will need to pull data from before registration
    def expand_slice(s_in, delta):
        if delta > 0:
            return slice(s_in.start, s_in.stop + delta, s_in.step)
        else:
            return slice(s_in.start + delta, s_in.stop, s_in.step)

    # Helper function to form slices we will need to pull data from after registration
    def get_final_slice(s_in, delta):
        if delta > 0:
            return slice(0, s_in.stop - s_in.start, s_in.step)
        else:
            return slice(-delta, s_in.stop - s_in.start - delta, s_in.step)

    n_slice_dims = len(image_slice)

    # Make sure user specified a standard slice
    if not is_standard_slice(image_slice):
        raise(NotImplementedError('Only an image_slice with non-negative start and stop values is currently supported.'))

    # Determine the rounded number of pixels we will shift - we will shift by the exact shift but
    # use the rounded value for determining the region of data that we need to pull before registration
    shift = t.affine[0:2, 2]
    rounded_shift = (np.ceil(np.abs(shift))*np.sign(shift)).astype(int)

    # Make sure the requested slice will have valid data in it after registration
    if len(image_shape) > 2:
        image_shape = image_shape[1:]
    valid_window = get_valid_translated_image_window(-1*shift, image_shape) # Account for sign convention on shifts

    if n_slice_dims < 3:
        good_window = slice_contains(image_slice, valid_window)
    else:
        good_window = slice_contains(image_slice[1:], valid_window)
    if not good_window:
        raise(ValueError('Requested slice contains bad data after registration. Valid window is: ' + str(valid_window)))

    # Form the slices we pull data from before and after registration
    if n_slice_dims < 3:
        orig_slice = tuple([expand_slice(image_slice[i], rounded_shift[i]) for i in range(n_slice_dims)])
        final_slice = tuple([get_final_slice(image_slice[i], rounded_shift[i]) for i in range(n_slice_dims)])
    else:
        orig_slice = tuple([image_slice[0], expand_slice(image_slice[1], rounded_shift[0]),
                            expand_slice(image_slice[2], rounded_shift[1])])
        final_slice = tuple([slice(0, image_slice[0].stop - image_slice[0].start + 1, image_slice[0].step),
                             get_final_slice(image_slice[1], rounded_shift[0]),
                             get_final_slice(image_slice[2], rounded_shift[1])])

    # Perform the registration
    orig_img = get_image_data(image, img_slice=orig_slice, h5_data_group=h5_data_group)
    reg_img = apply_2d_dipy_affine_transform(orig_img, t)

    # Now return the requested slice of the registered image
    return reg_img[final_slice]


def get_processed_image_data(images: list, func: types.FunctionType = None, img_slice = slice(None, None, None),
                             t_dict: dict = None, func_args: list = None, h5_data_group='default',
                             sc: pyspark.SparkContext = None) -> list:
    """ Gets processed image data for multiple images.
    
    This is a wrapper that allows retrieving images from files or from numpy arrays,
    applying light processing independently to each image and returning the result.

    Images can be optionally registered before apply preprocessing.
    
    Args:
        images: A list of images.  Each entry is either a numpy array or a path to an image file.
        
        func: A function to apply to each image.  If none, images will be returned unaltered.  Should accept input
        of the form func(image: np.ndarray, **keyword_args)

        img_slice: The slice of each image that should be returned before any processing is applied.  If registration is
        applies (see t_dict), then the slice coordinates are for images after registration.

        t_dict: A dictionary with information for performing image registration as images are loaded.  If set to None,
        no image registration will be performed.  t_dict should have two fields:

            transforms: A list of registration transforms to apply to the images as they are being read in.  If none,
            no registration will be applied.

            image_shape: This is the shape of the original images being read in.

        func_args: A list of extra keyword arguments to pass to the function for each image.  If None, no arguments
        will be passed.

        h5_data_group: The hdfs group holding image data in h5 files.
        
        sc: An optional pySpark.SparkContext object to use in speeding up reading of images.
        
    Returns: The processed image data as a list.  Each processed image is an entry in the list. 
    """

    n_images = len(images)
    do_reg = t_dict is not None

    if do_reg:
        img_transforms = t_dict['transforms']
        img_full_shape = t_dict['image_shape']
    else:
        img_transforms = [None]*n_images
        img_full_shape = None

    if func is None:
        def func(x):
            return x

    if func_args is None:
        func_args = [dict()]*n_images

    images_w_transforms_and_args = zip(images, img_transforms, func_args)

    if sc is None:
        return [func(get_reg_image_data(i_t[0], image_slice=img_slice, image_shape=img_full_shape,
                                        t=i_t[1], h5_data_group=h5_data_group), **i_t[2])
                for i_t in images_w_transforms_and_args]
    else:
        def _process_img(input):
            img = input[0]
            t = input[1]
            args = input[2]
            return func(get_reg_image_data(img, image_slice=img_slice, image_shape=img_full_shape,
                                           t=t, h5_data_group=h5_data_group), **args)
        return sc.parallelize(images_w_transforms_and_args).map(_process_img).collect()


def write_planes_to_files(planes: np.ndarray, files: list,
                          base_planes_dir: pathlib.Path, plane_suffix: str='plane',
                          skip_existing_files=False, sc: pyspark.SparkContext=None,
                          h5_data_group='data') -> list:
    """ Extracts one or more planes from image files, writing planes to seperate files.

    Args:
        planes: An array of indices of planes to extract.

        files: A list of original image files to pull planes from.

        base_planes_dir: The base directory to save plane files into.  Under this folder, seperate subfolders
        will be saved for each plane.

        plane_suffix: The suffix to append to the file name to indicate the file contains just one plane.

        skip_existing_files: If true, if a file for an extracted plane is found, it will not be overwritten.
        If false, then errors will be thrown if files for extracted planes are found to already exist. Setting
        this to true can be helpful if there is a need to run this function a second time to recover from an
        error.

        sc: An optional spark context to use to write files in parallel.

        h5_data_group: The h5_data_group that original images are stored under if reading in .h5 files.

    Returns:
        A list of the directories that images for each plane are saved into.
    """
    if not os.path.exists(base_planes_dir):
        os.makedirs(base_planes_dir)

    plane_dirs = []
    for plane in planes:
        plane_dir = base_planes_dir / (plane_suffix + str(plane))
        if not os.path.exists(plane_dir):
            os.makedirs(plane_dir)
        plane_dirs.append(plane_dir)

    if sc is None:
        for file in files:
            write_planes_for_one_file(file, planes, plane_dirs, '_' + plane_suffix, skip_existing_files,
                                      h5_data_group=h5_data_group)
    else:
        def write_plane_wrapper(file):
            write_planes_for_one_file(file, planes, plane_dirs, '_' + plane_suffix, skip_existing_files,
                                      h5_data_group=h5_data_group)
        sc.parallelize(files).foreach(write_plane_wrapper)

    return plane_dirs


def write_planes_for_one_file(file: pathlib.Path, planes: np.ndarray, plane_dirs: list,
                              plane_suffix: str='plane', skip_existing_files=False,
                              h5_data_group='default'):
    """ Writes specified planes from a 3d image file to separate .h5 files.

    The new files will have the same name as the original with an added suffix to indicate they contain
    just one plane.

    Args:
        planes: An array of indices of planes to extract.

        file: The original image file to pull planes from.

        plane_dirs: List of directories to save the file for each file into.

        plane_suffix: The suffix to append to the file name to indicate the file contains just one plane.

        skip_existing_files: If true, if a file for an extracted plane is found, it will not be overwritten.
        If false, then errors will be thrown if files for extracted planes are found to already exist. Setting
        this to true can be helpful if there is a need to run this function a second time to recover from an
        error.

        h5_data_group: The h5_data_group that original images are stored under if reading in .h5 files.

    """
    # Create names of the files the planes will be saved into
    new_file_name = file.name
    suffix_len = len(file.suffix)
    new_file_name = new_file_name[0:-suffix_len]
    plane_file_paths = [plane_dirs[i] / (new_file_name + plane_suffix + str(planes[i]) + '.h5')
                        for i in range(len(plane_dirs))]

    # Check if our files exist
    n_planes = len(planes)
    existing_plane_files = np.empty(n_planes, np.bool)
    some_plane_files_exist = False
    all_plane_files_exist = True
    for i, plane_file_path in enumerate(plane_file_paths):
        plane_file_exists = os.path.exists(plane_file_path)
        existing_plane_files[i] = plane_file_exists
        some_plane_files_exist = some_plane_files_exist or plane_file_exists
        all_plane_files_exist = all_plane_files_exist and plane_file_exists
    # Throw in an error if appropriate
    if some_plane_files_exist and not skip_existing_files:
        raise (RuntimeError('Files for extracted planes already exist for 3d image file ' + str(file)))

    # Write all planes that we need to to file
    if not all_plane_files_exist:
        image_3d = read_img_file(file, h5_data_group=h5_data_group)

        # Write planes to file
        for i, plane_file_path in enumerate(plane_file_paths):
            data_in_plane = np.expand_dims(image_3d[planes[i], :, :], 0)
            if not existing_plane_files[i]:
                with h5py.File(plane_file_path, 'w') as new_file:
                    new_file.create_dataset('data', data_in_plane.shape, data_in_plane.dtype, data_in_plane)

