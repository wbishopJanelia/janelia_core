""" Tools for reading in experimental data from labs at Janelia.

 William Bishop
 bishopw@hhmi.org

"""

import glob
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


def read_img_file(f_name: pathlib.Path, h5_data_group: str = 'default') -> np.ndarray:
    """ Returns a numpy array with data from one image file.
    
    The following file types are supported:
        1) .klb
        2) .h5
    
    Args:
        f_name: File name as pathlib.Path object

        h5_data_group: The group in a h5 file containing the image data (If reading in h5 files).
        
    Returns: 
        An numpy.ndarray object with the image data

    Raises:
        TypeError: If f_name is not a pathlib.Path object
        ValueError: If f_name does not have a supported extension

    """
    if not isinstance(f_name, pathlib.Path):
        raise TypeError('f_name is not a pathlib.Path object.')

    ext = f_name.suffix
    if ext == '.klb':
        return pyklb.readfull(str(f_name))  # pyklb requires string input
    if ext == '.h5':
        with h5py.File(f_name) as f:
            return f[h5_data_group][:]
    else:
        raise ValueError('File is a ' + ext + ' file, which is not currently supported.')


def process_wave_video_data_frame(datafame: pandas.DataFrame) -> list:
    """ Produces compact annotations of waves from annotations loaded from a csv file.

    This function is fairly rigidly written for annotations for the current Keller lab
    format.

    Args:
        dataframe: A dataframe object loaded from a csv file.  This can be achieved by:
            dataframe = pandas.read_csv(csv_file)

    Returns:
        annots: A list.  Each entry is a wave annotation.  When reading in the original
        annotations, if an original annotation had multiple types listed for the same wave,
        that annotation will be ignored.  Each entry in annots is a dictionary with the entries:
           s: The start index of the wave (indices are 0 based, and index the original images)
           e: The end index of the wave
           type: A string indicating what type of wave it is
           value: The value for the wave annotation in the original csv value
    """
    key_map = {'forward': 'forward wave',
               'backward': 'Backward wave',
               'hunch': 'Hunch',
               'turn': 'Turn',
               'other': 'Other'}

    def process_row(i, r):
        typ = None
        vl = None
        for k in key_map:
            cur_vl = r[key_map[k]]
            if pandas.notna(cur_vl):
                if typ is not None:
                    warn(RuntimeWarning('Found row with multiple types: Skipping row ' + str(i)))
                    return None
                typ = k
                vl = cur_vl

        return {'s': r['Start'] - 1, 'e': r['End'] - 1, 'type': typ, 'value': vl}

    annots = [process_row(*r) for r in dataframe.iterrows()]

    none_inds = np.where([a is None for a in annots])[0]
    for i in sorted(none_inds, reverse=True):
        del annots[i]

    return annots


def process_stimulus_inds(dataframe: pandas.DataFrame) -> list:
    """ Processes stimulus annotations loaded from a csv file.

    This function expects no header in the csv file and each line to contain
    two values in the order start, end giving the start and end index of the
    stimulus.  This function expects indexing *to be 1 based* and will convert to
    0 based when it produces output.

    Args:
        dataframe: A dataframe object loaded from a csv file.  This can be achieved by:
            dataframe = pandas.read_csv(csv_file)

    Returns:
        annots: A list.  Each entry is a stimulus event represented as a dictionary with the
        keys:
            s: the start index
            e: the end index
            type: A string which will always be 'stimulus'
    """
    stim_inds = list()
    for row_ind, row in dataframe.iterrows():
        stim_inds.append({'s': row['s'] - 1, 'e': row['e'] - 1, 'type': 'stimulus'})
    return stim_inds


