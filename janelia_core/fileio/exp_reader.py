""" Tools for reading in experimental data from labs at Janelia.

 William Bishop
 bishopw@hhmi.org

"""

import numpy
import pathlib

import dask
from dask.array import from_delayed, stack
from dask.delayed import delayed
import h5py

import pyklb


def img_files_to_dask_array(f_names: list, h5_data_group: str = 'default') -> dask.array.core.Array:
    """Returns a dask array object with stacked data from multiple image files.

    Each image must have data of the same size.

    Args:
        f_names: A list of file names as pathlib.Path objects

        h5_data_group: The group in a h5 file containing the image data (If reading in h5 files).

    Returns:
        A dask.array.core.Array object with stacked data from multiple image files.
        Stacking is along the first dimension so if data is a returned array of 3-d
        volumetric data data[i, :, :, :] would correspond to the i^th volume.
    """
    first_image = read_img_file(f_names[0])
    shape = first_image.shape
    dtype = first_image.dtype

    return stack([from_delayed(delayed(read_img_file, pure=True)(f, h5_data_group), shape, dtype) for f in f_names])


def read_img_file(f_name: pathlib.Path, h5_data_group: str = 'default') -> numpy.ndarray:
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
