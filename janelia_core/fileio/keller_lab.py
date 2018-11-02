""" Tools for reading in Keller lab experimental data.

    William Bishop
    bishopw@hhmi.org
"""

import glob
import numpy as np
import os
import pathlib
from shutil import copyfile
from xml.etree import ElementTree as ET

import janelia_core.dataprocessing.dataset
import janelia_core.dataprocessing.dataset as dataset
import janelia_core.fileio.exp_reader as exp_reader

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

    This function will read all experimental data into a DataSet object.  Imaging data will be represented as a
    dask.array object.

    Args:
        image_folder: The folder containing the images (see above) as a pathlib.Path object

        metadata_folder: The folder containing metadata (see above) as a pathlib.Path object

        image_ext: String to use when searching for files with the extension for image files.

        metadata_file: Name of file containing metadata.

        time_stamps_file: Name of file containing time points each image volume was acquired at

        verbose: True if progress updates should be printed to screen

    Returns:
        A DataSet object representing the experiment.  The data dictionary will have one entry with the
        key 'images' containing a dask.array object with the image data. The metadata for the experiment will
        have an entry with the key 'image_names' containing the image names in an order corresponding to how image
        data is order in data['images'].

    Raises:
        RuntimeError: If the number of images found differs from the number of time stamps in the time_stamps_file.

    """

    metadata = read_metadata(metadata_folder / metadata_file)
    time_stamps = read_time_stamps(metadata_folder / time_stamps_file)
    image_names, dask_array = read_images(image_folder, image_ext, verbose)

    n_images = dask_array.shape[0]
    n_time_stamps = time_stamps.size
    if n_images != n_time_stamps:
        raise(RuntimeError('Found ' + str(n_images) + ' image files but ' + str(n_time_stamps) + ' time stamps.'))

    data_dict = {'images': dask_array}

    metadata['image_names'] = image_names

    return dataset.DataSet(data_dict, time_stamps, metadata)


def read_metadata(metadata_file: pathlib.Path) -> dict:
    """Function to read in Keller lab metadata stored in xml files.

    This function does some very light processing of the data in the xml files.

    Note: Keller lab metadata is stored as attributes of XML elements.

    Args:
        metadata_file: The xml file with metadata

    Returns:
        A dictionary of metadata.

    Raises:
        RuntimeError: If xml elements produce dictionaries which have more than one key (This function
            currently only handles simple xml structure.)

        RuntimeError: If xml elments have the same tag and attributes (This function currently assumes this
            can't happen).

    """

    # First replace '&' characters with 'and' so xml will parse
    with open(metadata_file) as xml_file:
        metadata_string = xml_file.read().replace('&', ' and ')

    # Read in xml data
    metadata_root = ET.fromstring(metadata_string)

    # Now convert DOM tree to dictionary
    metadata = dict()

    # See all the tags we have and create sub-dictionaries for each tag
    all_tags = [c.tag for c in metadata_root]
    unique_tags = set(all_tags)
    for t in unique_tags:
        metadata[t] = dict()

    for c in metadata_root:
        element_tag = c.tag
        element_dict = c.attrib
        element_keys = list(element_dict.keys())

        if len(element_keys) != 1:
            raise(RuntimeError('Found xml element with multiple attributes.'))

        element_key = element_keys[0]
        tag_dict = metadata[element_tag]
        if element_key in tag_dict.keys():
            raise(RuntimeError('Found xml elments with the same tags and attributes.'))
        tag_dict[element_key] = element_dict[element_key]

    return metadata


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


def read_images(image_folder: pathlib.Path, image_ext: str='.weightFused.TimeRegistration.klb', verbose=True) -> list:
    """Locates Keller lab image files and creates a dask.array object.

    Args:
        image_folder: The folder containing the images (see above) as a pathlib.Path object

        image_ext: String to use when searching for files with the extension for image files.

        verbose: True if progress updates should be printed to screen

    Returns:
        Returns a list.  The first element of the list is a list of file names. The second element of the
        list is a dask.array object.  The ordering of data in the array matches the ordering of file names so
        that list[0][i] is the filename for the data in list[1][i,:,:,:].  Data is ordered in chronological order.

    Raises:
        RuntimeError: If no image files are located.
    """
    if verbose:
        print('Searching for image files...')
    img_files = glob.glob(str(image_folder / '*' / ('*' + image_ext)))
    n_img_files = len(img_files)

    if n_img_files == 0:
        raise(RuntimeError('Unable to find any ' + image_ext + ' files in the subfolders of ' + str(image_folder)))

    # Make sure our image files are sorted
    files_as_paths = [pathlib.Path(f) for f in img_files]
    smp_inds = np.asarray([int(f.name[IMAGE_NAME_SMP_START_IND:IMAGE_NAME_SMP_END_IND]) for f in files_as_paths])
    sort_order = np.argsort(smp_inds)
    sorted_files_as_paths = [files_as_paths[sort_order[i]] for i in sort_order]

    if verbose:
        print('Creating dask.array object from ' + str(n_img_files) + ' image files.')
    dask_array = exp_reader.img_files_to_dask_array(sorted_files_as_paths)

    if verbose:
        print('Done reading in image files.')
    return [sorted_files_as_paths, dask_array]


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

