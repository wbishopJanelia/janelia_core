""" Tools for reading in Keller lab experimental data.

    William Bishop
    bishopw@hhmi.org
"""

import glob
import numpy as np
import os
import pathlib
from shutil import copyfile

import h5py
import pyspark

import janelia_core.dataprocessing.dataset
import janelia_core.dataprocessing.dataset as dataset
from janelia_core.fileio.exp_reader import  read_img_file
from janelia_core.fileio.exp_reader import read_imaging_metadata
from janelia_core.fileio.exp_reader import find_images

# Indices in image file names indicating which sample they correspond to
IMAGE_NAME_SMP_START_IND = 8
IMAGE_NAME_SMP_END_IND = 14


def read_exp(image_folder: pathlib.Path, metadata_folder: pathlib.Path,
             image_ext: str='.weightFused.TimeRegistration.klb', metadata_file: str= 'ch0.xml',
             time_stamps_file:str = 'TM elapsed times.txt', verbose=True) -> janelia_core.dataprocessing.dataset.DataSet:
    """Function to read in data from a Keller lab VNC experiment.

    This function assumes all the imaging files are stored the one image_folder in the form:
        image_folder/TM[smp_number]/*.image_ext

    It also assumes the file with the metadata is stored at:
        metadata_folder/meta_data_file

    and the time point data is stored at:
        metadata_folder/time_stamps_file

    This function will read all experimental data into a DataSet object.

    Args:
        image_folder: The folder containing the images (see above) as a pathlib.Path object

        metadata_folder: The folder containing metadata (see above) as a pathlib.Path object

        image_ext: String to use when searching for files with the extension for image files.

        metadata_file: Name of file containing metadata.

        time_stamps_file: Name of file containing time points each image volume was acquired at

        verbose: True if progress updates should be printed to screen

    Returns:
        A DataSet object representing the experiment.  The data dictionary will have one entry with the
        key 'imgs' containing paths to each image file in the dataset.

    Raises:
        RuntimeError: If the number of images found differs from the number of time stamps in the time_stamps_file.

    """

    metadata = read_imaging_metadata(metadata_folder / metadata_file)
    time_stamps = read_time_stamps(metadata_folder / time_stamps_file)
    image_names_sorted = find_images(image_folder, image_ext, image_folder_depth=1, verbose=verbose)

    n_images = len(image_names_sorted)
    n_time_stamps = time_stamps.size
    if n_images != n_time_stamps:
        raise(RuntimeError('Found ' + str(n_images) + ' image files but ' + str(n_time_stamps) + ' time stamps.'))

    im_dict = {'ts': time_stamps, 'vls': image_names_sorted}
    data_dict = {'imgs': im_dict}

    return dataset.DataSet(data_dict, metadata)


def read_time_stamps(time_stamp_file: pathlib.Path) -> np.ndarray:
    """Reads in Keller lab text files containing time stamps images were acquired at.

    This function currently assumes the first line of the text file is a date and time and
    the second line contains the time stamps.

    Args:
        time_stamp_file: The time stamps file to read in.

    Returns:
        A 1-d numpy ndarray with the time time stamps

    """
    time_stamp_line = 2

    with open(time_stamp_file, 'r') as f:
        txt_lines = f.readlines()

    text_vls = txt_lines[time_stamp_line].split('\t')
    time_stamps = np.asarray([float(i) for i in text_vls])

    if not(np.isclose(time_stamps[0], 0)):
        raise(RuntimeError('First recovered time stamp is not 0.'))

    return time_stamps


def copy_exp(image_folder: pathlib.Path, metadata_folder: pathlib.Path, dest_folder: pathlib.Path,
             image_ext: str='.weightFused.TimeRegistration.klb', metadata_file: str='ch0.xml',
             time_stamps_file:str = 'TM elapsed times.txt', verbose=True, update_int=10):
    """Function to copy data for a Keller lab VNC experiment from one set of locations to another.

        This function can be useful for copying files from a network drive to location storage.

        Images, metadata and timestamp data are copied to the same destination folder (even
        though they may be in different source folders).

        This function assumes all the source imaging files are stored the one image_folder in the form:
            image_folder/TM[smp_number]/*.image_ext

        It also assumes the file with the metadata is stored at:
            metadata_folder/meta_data_file

        and the time point data is stored at:
            metadata_folder/time_stamps_file

        Args:
            image_folder: The folder containing the images (see above) as a pathlib.Path object

            metadata_folder: The folder containing metadata (see above) as a pathlib.Path object

            dest_folder: The destination folder to copy to as a pathlib.Path object

            image_ext: String to use when searching for files with the extension for image files.

            metadata_file: Name of file containing metadata.

            time_stamps_file: Name of file containing time points each image volume was acquired at

            verbose: True if progress updates should be printed to screen

            update_int: How many files should be copied before updating user on status

        Raises:
            RuntimeError: If destination metadata or images folders already exist
    """

    # Copy metadata
    if verbose:
        print('Copying metadata and time stamp files...')
    new_metadata_folder = pathlib.Path(dest_folder / 'metadata')
    if os.path.isdir(new_metadata_folder):
        raise(RuntimeError('The directory ' + str(new_metadata_folder) + ' already exists.'))
    os.mkdir(new_metadata_folder)
    copyfile(metadata_folder / metadata_file, new_metadata_folder / metadata_file)
    copyfile(metadata_folder / time_stamps_file, new_metadata_folder / time_stamps_file)
    if verbose:
        print('Done.')

    # Copy images
    if verbose:
        print('Searching for image files...')
    img_files = glob.glob(str(image_folder / '*' / ('*' + image_ext)))
    n_img_files = len(img_files)

    if n_img_files == 0:
        raise(RuntimeError('Unable to find any ' + image_ext + ' files in the subfolders of ' + str(image_folder)))
    if verbose:
        print('Done.')
        print('Copying image files...')

    new_images_folder = pathlib.Path(dest_folder / 'images')
    if os.path.isdir(new_images_folder):
        raise (RuntimeError('The directory ' + str(new_images_folder) + ' already exists.'))
    os.mkdir(new_images_folder)

    for i, f in enumerate(img_files):
        source_file = pathlib.Path(f)
        containing_folder = source_file.parts[-2]
        new_containing_folder = new_images_folder / containing_folder
        os.mkdir(new_containing_folder)
        dest_file =  new_containing_folder / source_file.name
        copyfile(source_file, dest_file)
        if verbose and (i + 1) % update_int == 0:
            print('Done copying file ' + str(i + 1) + ' of ' + str(n_img_files) + '.')

    if verbose:
        print('Done.')
        print('All files copied successfully.')


def write_planes_to_file(orig_files: list, plane: int, target_dir: pathlib.Path,
                        plane_suffix: str = '_p', sc: pyspark.SparkContext = None) -> list:
    """ Function to read in a list of .klb files, extract one plane from each, and right these to h5 files.

    Each plane will be stored in a separate file. The new file names will be the same as the first with a suffix
    added of the form [plane_suffix]#, where # is the plane number.

    To speed up computation, an optional SparkContext object can be provided.

    Args:

        orig_files: A list of files.  Each file should be a pathlib.Path object

        plane: The z-index of the plane to extract across files.

        target_dir: The directory to save the files into as a pathlib.Path object.  If this does not exist,
        it will be created.

        plane_suffix: The suffix to append to the file name to indicate the file contains just one plane.

        sc: A SparkContext option if spark should be used

    Returns: A list of the new files created, in the same order as orig_files.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    def extract_and_write(file):
        data_in_plane = np.expand_dims(read_img_file(file)[plane, :, :], 0)

        # Create the new file name
        new_file_name = file.name
        suffix_len = len(file.suffix)
        new_file_name = new_file_name[0:-suffix_len] + plane_suffix + str(plane) + '.h5'
        new_file_path = target_dir / new_file_name

        # Function to write the new file
        with h5py.File(new_file_path, 'w') as new_file:
            new_file.create_dataset('data', data_in_plane.shape, data_in_plane.dtype, data_in_plane)

        return new_file_path

    if sc is not None:
        return sc.parallelize(orig_files).map(extract_and_write).collect()
    else:
        new_files = []
        for f in orig_files:
            new_files.append(extract_and_write(f))
        return new_files
