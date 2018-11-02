""" Tools for reading in Ahrens lab experimental data.

    William Bishop
    bishopw@hhmi.org
"""

import glob
import pathlib

import numpy as np

import janelia_core.dataprocessing.dataset
import janelia_core.dataprocessing.dataset as dataset
from janelia_core.fileio.shared_lab import read_imaging_metadata
from janelia_core.fileio.shared_lab import read_images

# Constants for reading a stack frequency file
STACK_FREQ_STACK_FREQ_LINE = 0
STACK_FREQ_EXP_DURATION_LINE = 1
STACK_FREQ_N_IMAGES_LINE = 2


def read_exp(image_folder: pathlib.Path, ephys_folder: pathlib.Path = None,
             image_ext: str = '.h5', h5_data_group: str = 'default', metadata_file: str = 'ch0.xml',
             stack_freq_file: str = 'Stack_frequency.txt', ephys_file : str = 'rawdata.mat',
             verbose: bool = True) -> janelia_core.dataprocessing.dataset.DataSet:
    """Reads in Ahrens lab experimental data to a Dataset object.

    This function is currently developed for reading in two color glial data but can be generalized in
    the future.

    Args:
        image_folder: The folder holding the images, metadata file and stack frequency file.

        ephys_folder: The folder holder the ephys data.

        image_ext: The extension to use when looking for image files.

        h5_data_group: The data group in .h5 image files containing image data.

        metadata_file: The name of the .xml file holding metadata.

        stack_freq_file: The name of the file holding stack frequency information.

        ephys_file: The name of the file holding ephys data.  If this is None, no ephys data will be loaded.

        verbose: True if progress updates should be printed to screen.

    Returns:
        A Dataset object.  A DataSet object representing the experiment.  The data dictionary will have an entry 'imgs'
        'imgs' containing the image data. If ephys data was available, an entry 'ephys' will also contain the ephys
        data.  The metadata for the experiment will have an entry with the key 'image_names' containing the image names
        in an order corresponding to how image data is order in the imgs dictionary.  It will also have an entry
        'stack_freq_info' with the information from the stack frequency file.
    """

    # Read in all of the raw data
    metadata = read_imaging_metadata(image_folder / metadata_file)

    stack_freq_info = read_stack_freq(image_folder / stack_freq_file)

    image_names, dask_array = read_images(image_folder, image_ext, image_folder_depth=0,
                                          h5_data_group=h5_data_group, verbose=verbose)
    n_images = dask_array.shape[0]
    time_stamps = np.asarray([float(i / stack_freq_info['smp_freq']) for i in range(n_images)])

    if ephys_file is not None:
        raise(NotImplementedError('Functionality to read in ephys data has not yet been implemented.'))

    # Check to make we found the right number of images
    if n_images != stack_freq_info['n_images']:
        raise (RuntimeError('Found ' + str(n_images) + ' image files stack frequency file specified ' +
                            str(stack_freq_info['n_images']) + ' time stamps.'))

    # Create an instance of Dataset
    im_dict = {'ts': time_stamps, 'vls': dask_array}
    data_dict = {'imgs': im_dict}

    metadata['image_names'] = image_names
    metadata['stack_freq_info'] = stack_freq_info

    return dataset.DataSet(data_dict, metadata)


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
