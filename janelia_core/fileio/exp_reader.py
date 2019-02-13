""" Tools for reading in experimental data from labs at Janelia.

 William Bishop
 bishopw@hhmi.org

"""

import glob
import os
import pathlib
import re
from xml.etree import ElementTree as ET
import warnings

import h5py
import numpy as np

try:
    import pyklb
except ModuleNotFoundError as error:
    warnings.warn('Unable to locate pyklb module.  Will not be able to read in .klb files.')

# Regular expression to use for parsing sample numbers from image file names
IMG_SMP_NUM_FILE_NAME_REG_EXP = r'(.*)(TM)([0123456789]*)_'


def find_images(image_folder: pathlib.Path, image_ext: str, image_folder_depth: int = 0, verbose=True) -> list:
    """ Locates image files, returning paths to files in a sorted list.

    Image files are sorted by sample number.

    Args:
        image_folder: The folder directly containing images or containing subfolders which contain the images.

        image_ext: String to use when searching for files with the extension for image files.

        image_folder_depth: The number of layers of subfolders under image_folder to look to find the image files. If
            this is 0, then the images are directly under image_folder.

        verbose: True if progress updates should be printed to screen

    Returns: A list of image files as pathlib.Path objects.

    """
    if verbose:
        print('Searching for image files...')

    image_glob_path = image_folder
    cur_depth = 0
    while cur_depth < image_folder_depth:
        image_glob_path = image_glob_path / '*'
        cur_depth += 1
    image_glob_path = image_glob_path / ('*' + image_ext)

    img_files = glob.glob(str(image_glob_path))
    n_img_files = len(img_files)

    if n_img_files == 0:
        raise(RuntimeError('Unable to find any ' + image_ext + ' files under ' + str(image_folder)))

    # Make sure our image files are sorted
    files_as_paths = [pathlib.Path(f) for f in img_files]
    smp_inds = np.asarray([int(re.match(IMG_SMP_NUM_FILE_NAME_REG_EXP, f.name).group(3)) for f in files_as_paths])
    sort_order = np.argsort(smp_inds)

    print('Found ' + str(n_img_files) + ' images.')

    return [files_as_paths[i] for i in sort_order]


def read_imaging_metadata(metadata_file: pathlib.Path) -> dict:
    """Function to read in imaging metadata stored in xml files as dictionaries.

    This function does some very light processing of the data in the xml files.

    Note: Imaging metadata is stored as attributes of XML elements.  The attribute names will
    be the keys in the crated dictionaries. Attribute names will be appended with '_#' when
    needed to avoid duplicate keys in the dictionary.

    Args:
        metadata_file: The xml file with metadata

    Returns:
        A dictionary of metadata.

    Raises:
        RuntimeError: If xml elements produce dictionaries which have more than one key (This function
            currently only handles xml tags with single attributes.)
    """

    # First replace '&' characters with 'and' so xml will parse
    with open(metadata_file, 'r') as xml_file:
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

        # If we haven't added an element to this tag dictionary with the same key
        # before, we just use the original element_key (which is the attribute name
        # in the xml).  If not, we add a number to the key to make it unique.
        new_element_key = element_key = element_keys[0]
        tag_dict = metadata[element_tag]
        if new_element_key in tag_dict.keys():
            i = 2
            new_element_key = element_key + '_' + str(i)
            while new_element_key in tag_dict.keys():
                i = i + 1
                new_element_key = element_key + '_' + str(i)

        tag_dict[new_element_key] = element_dict[element_key]

    return metadata


def read_img_file(f_name: pathlib.Path, img_slice: slice = slice(None, None, None),
                  h5_data_group: str = 'default') -> np.ndarray:
    """ Returns a numpy array with data from one image file.
    
    The following file types are supported:
        1) .klb
        2) .h5
    
    Args:
        f_name: File name as pathlib.Path object

        img_slice: The slice of the image that should be returned

        h5_data_group: The group in a h5 file containing the image data (If reading in h5 files).
        
    Returns: 
        An numpy.ndarray object with the image data

    Raises:
        TypeError: If f_name is not a pathlib.Path object
        ValueError: If f_name does not have a supported extension
        NotImplementedError: If passing in a non-default slice object while reading a .klb file

    """
    if not isinstance(f_name, pathlib.Path):
        raise TypeError('f_name is not a pathlib.Path object.')

    ext = f_name.suffix
    if ext == '.klb':
        if img_slice != slice(None, None, None):
            raise(NotImplementedError('Slicing while reading in .klb files is not currently supported.'))
        return pyklb.readfull(str(f_name))  # pyklb requires string input
    if ext == '.h5':
        with h5py.File(f_name) as f:
            return f[h5_data_group][img_slice]
    else:
        raise ValueError('File is a ' + ext + ' file, which is not currently supported.')

