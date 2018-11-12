""" Tools for reading in Ahrens lab experimental data.

    William Bishop
    bishopw@hhmi.org
"""

import glob
import pathlib

import h5py
import numpy as np

import janelia_core.dataprocessing.dataset
import janelia_core.dataprocessing.dataset as dataset
from janelia_core.fileio.exp_reader import read_imaging_metadata
from janelia_core.fileio.exp_reader import find_images

# Constants for reading a stack frequency file
STACK_FREQ_STACK_FREQ_LINE = 0
STACK_FREQ_EXP_DURATION_LINE = 1
STACK_FREQ_N_IMAGES_LINE = 2


def read_exp(image_folder: pathlib.Path, ephys_folder: pathlib.Path = None, ephys_file: str = 'frame_swim.mat',
             ephys_var_name: str = 'frame_swim', image_ext: str = '.h5', metadata_file: str = 'ch0.xml',
             stack_freq_file: str = 'Stack_frequency.txt', verbose: bool = True) -> janelia_core.dataprocessing.dataset.DataSet:
    """Reads in Ahrens lab experimental data to a Dataset object.

    Args:
        image_folder: The folder holding the images, metadata file and stack frequency file.

        ephys_folder: The folder holding the ephys data.

        ephys_file: The name of the file holding ephys data.  If this is None, no ephys data will be loaded.

        ephys_var_name: The variable name holding ephys data in ephys_file.

        image_ext: The extension to use when looking for image files.

        metadata_file: The name of the .xml file holding metadata.

        stack_freq_file: The name of the file holding stack frequency information.

        verbose: True if progress updates should be printed to screen.

    Returns:
        A Dataset object.  A DataSet object representing the experiment.  The data dictionary will have an entry 'imgs'
        containing the file names for the images. If ephys data was available, an entry 'ephys' will also contain
        the ephys data.  The metadata for the experiment will have an entry 'stack_freq_info' with the information from
        the stack frequency file.
    """

    # Read in all of the raw data
    metadata = read_imaging_metadata(image_folder / metadata_file)

    stack_freq_info = read_stack_freq(image_folder / stack_freq_file)

    image_names_sorted = find_images(image_folder, image_ext, image_folder_depth=0, verbose=verbose)

    n_images = len(image_names_sorted)
    time_stamps = np.asarray([float(i / stack_freq_info['smp_freq']) for i in range(n_images)])

    if ephys_file is not None:
        ephys_data = read_ephys_data(ephys_folder / ephys_file, ephys_var_name, verbose=verbose)
        n_ephys_smps = ephys_data.shape[0]
        if n_ephys_smps != n_images:
            raise (RuntimeError('Found ' + str(n_images) + ' image files but ' + str(n_ephys_smps) + ' ephys data points.'))
        ephys_dict = {'ts': time_stamps, 'vls': ephys_data}

    # Check to make we found the right number of images
    if n_images != stack_freq_info['n_images']:
        raise (RuntimeError('Found ' + str(n_images) + ' image files but stack frequency file specified ' +
                            str(stack_freq_info['n_images']) + ' time stamps.'))

    # Create an instance of Dataset
    im_dict = {'ts': time_stamps, 'vls': image_names_sorted}
    data_dict = {'imgs': im_dict}
    if ephys_file is not None:
        data_dict['ephys'] = ephys_dict

    metadata['stack_freq_info'] = stack_freq_info

    return dataset.DataSet(data_dict, metadata)


def read_ephys_data(ephys_file: pathlib.Path, var_name: str = 'frame_swim', verbose: bool = True):
    """ Reads in electophysiological data for an Ahrens lab experiment.

    Args:
        ephys_file: The path to the .mat file holding the data

        var_name: The name of the variable in the .mat file holding the
        ephys_data

    Returns:
        A numpy.ndarray with the data
    """
    if verbose:
        print('Reading ephys data.')

    with h5py.File(ephys_file) as f:
        data = f[var_name][:]
        data = data.T
        return data


def read_stack_freq(stack_freq_file: pathlib.Path):
    """ Reads in stack frequency inforation from file.

    Args:
        stack_freq_file: The file with stack frequency information.

    Returns:
        A dictionary with the stack frequency information.
    """

    with open(stack_freq_file, 'r') as f:
        txt_lines = f.readlines()

        smp_freq = float(txt_lines[STACK_FREQ_STACK_FREQ_LINE])
        exp_duration = float(txt_lines[STACK_FREQ_EXP_DURATION_LINE])
        n_images = int(txt_lines[STACK_FREQ_N_IMAGES_LINE])

        return {'smp_freq': smp_freq, 'exp_duration' : exp_duration, 'n_images' : n_images}


def read_seperated_exp(image_folders: list, image_labels: list, metadata_folder: pathlib.Path, ephys_folder: pathlib.Path = None,
             ephys_file : str = 'frame_swim.mat', ephys_var_name: str = 'frame_swim', image_ext: str = '.h5',
             metadata_file: str = 'ch0.xml', stack_freq_file: str = 'Stack_frequency.txt',
             verbose: bool = True) -> janelia_core.dataprocessing.dataset.DataSet:
    """Reads in Ahrens lab experimental data after images have been split by color.

    Args:
        image_folders: List of folders holding split data (e.g., one folder for images of neurons and one for glia).

        metadata_folder: Folder containing experiment metadata.

        image_labels: List of string labels for each series of images in image_folder.

        ephys_folder: The folder holding the ephys data.

        ephys_file: The name of the file holding ephys data. If this is None, no ephys data will be loaded.

        ephys_var_name: The variable name holding ephys data in ephys_file.

        image_ext: The extension to use when looking for image files.

        metadata_file: The name of the .xml file holding metadata.

        stack_freq_file: The name of the file holding stack frequency information.

        verbose: True if progress updates should be printed to screen.

    Returns:
        A Dataset object.  A DataSet object representing the experiment.  The data dictionary will have entries
        containing the file names for the images for each color. If ephys data was available, an entry 'ephys' will also
        contain the ephys data.  The metadata for the experiment will have an entry 'stack_freq_info' with the information from
        the stack frequency file.
    """

    # Read in all of the raw data
    metadata = read_imaging_metadata(metadata_folder / metadata_file)

    stack_freq_info = read_stack_freq(metadata_folder / stack_freq_file)

    image_names_sorted = []
    for image_folder in image_folders:
        cur_sorted_names = find_images(image_folder, image_ext, image_folder_depth=0, verbose=verbose)
        image_names_sorted.append(cur_sorted_names)

    n_series_one_images = len(image_names_sorted[0])
    for i, sorted_names in enumerate(image_names_sorted):
        n_cur_images = len(sorted_names)
        if n_series_one_images != n_cur_images:
            raise(RuntimeError('All image series must have the same number of images. Found ' + str(n_series_one_images) +
                               ' but ' + str(n_cur_images) + ' images for series ' + str(i) + '.'))

    time_stamps = np.asarray([float(i / stack_freq_info['smp_freq']) for i in range(n_series_one_images)])

    if ephys_file is not None:
        ephys_data = read_ephys_data(ephys_folder / ephys_file, ephys_var_name, verbose=verbose)
        n_ephys_smps = ephys_data.shape[0]
        if n_ephys_smps != n_series_one_images:
            raise (RuntimeError(
                'Found images for ' + str(n_series_one_images) + ' time points but ' + str(n_ephys_smps) + ' ephys data points.'))
        ephys_dict = {'ts': time_stamps, 'vls': ephys_data}

    # Check to make we found the number of image files expected from sampling frequency data
    if n_series_one_images != stack_freq_info['n_images']:
        raise (RuntimeError('Found ' + str(n_series_one_images) + ' image files but stack frequency file specified ' +
                            str(stack_freq_info['n_images']) + ' time stamps.'))

    # Create an instance of Dataset
    data_dict = {}
    for i, sorted_names in enumerate(image_names_sorted):
        im_dict = {'ts': time_stamps, 'vls': sorted_names}
        data_dict[image_labels[i]] = im_dict

    if ephys_file is not None:
        data_dict['ephys'] = ephys_dict

    metadata['stack_freq_info'] = stack_freq_info

    return dataset.DataSet(data_dict, metadata)
