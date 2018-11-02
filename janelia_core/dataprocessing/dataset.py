""" Defines dataset class.  In the future this may include more tools for working with datasets.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


class DataSet:
    """A class for holding a basic data set.

    Args:
        data: A dictionary.  Each entry contains a numpy.ndarray or dask.array object of data.  The arrays can be of
            varying sizes but in all cases, the fist dimension must correspond to time.  If data is none the data attribute
            of the created object will be an empty dictionary.

        time_pts: A 1-d numpy.ndarray array of time points for the data in the data dictionary. If time_pts is none the
            data attribute of the crated object will be a 0-dimensional numpy array

        metadata: A dictionary of metadata. If meta_data is None, the meta_data attribute of the created object will be
            a empty dictionary.
    """

    def __init__(self, data: dict=None, time_pts: np.ndarray=None, metadata: dict=None):
        if data is None:
            self.data = dict()
        else:
            self.data = data

        if time_pts is None:
            self.time_pts = np.empty(0)
        else:
            self.time_pts = time_pts

        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata


