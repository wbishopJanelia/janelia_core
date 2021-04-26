""" Contains tools for saving research results.

The tools here enable results to be saved and archived for later retrieval.

    William Bishop
    bishopw@hhmi.org
"""

import datetime
import os
import pathlib
from typing import Union

import h5py
import numpy as np

HDF5_TYPES = {'nparray': 'nparray',
              'dict': 'dict',
              'list': 'list'}



def append_ts(filename: str, no_underscores: bool = False) -> str:
    """ Appends a time stamp to a string.

    The primary use case for this file is taking a file name and appending a time stamp to
    it.  This is useful when multiple analyses are run and we want to save the results of
    each with a unique name.

    The time stamps will be of the format _<4 digit year>_<2 digit month>_<two digit day>_<2 digit military time hour>...
                                            _<2 digit minute>_<2 digit second>_<0 padded microsecond>

    Args:

        filename: The filename to append to

        no_underscores: If true, underscores in the date string above will be omitted.
    """

    data_str = '{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.datetime.now())

    if no_underscores:
        data_str = data_str.replace('_', '')

    return filename + '_' + data_str

def save_structured_hdf5(o: Union[np.ndarray, list, dict], f: pathlib.Path, name: str, overwrite: bool = False):
    """ Saves structured data to an hdf5 file.

    Args:
        o: The structured data to save, which can consist of nested dictionaries, lists and
        numpy arrays containing numeric data.

        f: A path to the file to save the data in.

        name: The name to save the data under

        overwrite: Overwrites existing file if it exists
    """

    if overwrite:
        if os.path.exists(f):
            os.remove(f)

    def _recursive_save(o, f, group, key):
        if isinstance(o, np.ndarray):
            with h5py.File(f, 'a') as f_h:
                dset = f_h.create_dataset(group, data=o)
                dset.attrs['type'] = HDF5_TYPES['nparray']
                dset.attrs['name'] = key
        elif isinstance(o, list):
            with h5py.File(f, 'a') as f_h:
                grp = f_h.create_group(group)
                grp.attrs['type'] = HDF5_TYPES['list']
                grp.attrs['name'] = key
            for v_i, v in enumerate(o):
                _recursive_save(v, f, group + '/list_' + str(v_i), v_i)
        elif isinstance(o, dict):
            with h5py.File(f, 'a') as f_h:
                grp = f_h.create_group(group)
                grp.attrs['type'] = HDF5_TYPES['dict']
                grp.attrs['name'] = key
            for k in o.keys():
                _recursive_save(o[k], f, group + '/' + str(k), k)

    _recursive_save(o, f, name, name)


def load_structured_hdf5(f: pathlib.Path):
    """ Loads data saved by save_structured_hdf5.

    Args:
        f: A path to the file with the saved data in it.

    """

    def _recursive_load(f, grp):
        with h5py.File(f, 'r') as f_h:
            obj = f_h[grp]
            obj_type = obj.attrs['type']

        if obj_type == HDF5_TYPES['nparray']:
            with h5py.File(f, 'r') as f_h:
                return f_h[grp][:]
        elif (obj_type == HDF5_TYPES['list']) or (obj_type == HDF5_TYPES['dict']):
            with h5py.File(f, 'r') as f_h:
                obj = f_h[grp]
                obj_keys = list(obj.keys())
                label_keys = [f_h[grp][k].attrs['name'] for k in obj_keys]
            if obj_type == HDF5_TYPES['list']:
                n_list_entries = len(label_keys)
                return_list = [None]*n_list_entries
                for i in range(n_list_entries):
                    return_list[label_keys[i]] = _recursive_load(f, grp + '/' + obj_keys[i])
                return return_list
            else:
                # obj_type is a dictionary
                return {label_k: _recursive_load(f, grp + '/' + k) for label_k, k in zip(label_keys, obj_keys)}
        else:
            raise(ValueError('Unrecogonized type: ' + obj_type))

    with h5py.File(f, 'r') as f_h:
        top_grp = list(f_h.keys())[0]

    return _recursive_load(f, top_grp)

