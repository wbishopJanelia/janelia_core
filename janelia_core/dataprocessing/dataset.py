""" Tools for working with imaging datasets.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


class DataSet:
    """A class for holding a basic data set.

    Args:
        ts_data: A dictionary of time series data.  Each entry in the dictionary is one set of data (with a user
            specified key).  Each set of data is itself stored in a dictionary with two entries.  The fist with
            the key 'ts' is a 1-d numpy.ndarray with timestamps.  The second 'vls' is a list of or numpy array of data
            for each point in ts. If data is none the data attribute of the created object will be an empty dictionary.

        metadata: A dictionary of metadata. If meta_data is None, the meta_data attribute of the created object will be
            a empty dictionary.
    """

    def __init__(self, ts_data: dict=None, metadata: dict=None):
        if ts_data is None:
            self.ts_data = dict()
        else:
            self.ts_data = ts_data

        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

    def has_ts_data(self) -> bool:
        """Returns true if any time series data has non-zero data points.
        """
        ts_keys = self.ts_data.keys()
        has_data = True
        for k in ts_keys:
            has_data = has_data and len(self.ts_data[k]['ts']) > 0
            if not has_data:
                break
        return has_data

    def select_time_range(self, start_time: float, end_time: float) -> 'DataSet':
        """Selects data in a time range and returns a new dataset Object.

        The start and end times are inclusive.

        Args:
            start_time: The start time of the selection.

            end_time: The end time of the selection.

        Returns:
            New dataset object containing just those time points requested.
        """

        ts_data_keys = self.ts_data.keys()
        sel_ts_data = {}
        for key in ts_data_keys:
            sel_inds = np.flatnonzero(np.logical_and(self.ts_data[key]['ts'] >= start_time, self.ts_data[key]['ts'] <= end_time))
            sel_ts_data[key] = self.select_ts_data(key, sel_inds)
        return DataSet(sel_ts_data, self.metadata)

    def select_ts_data(self, key:str, sel_inds: np.ndarray) -> dict:
        """ Selects data by index in a given dictionary of ts_data.

        This function will allow selection to work seamlessly no matter the
        type of 'vls' in the dictionary data is being pulled from.

        Args:
            key: The key of the ts_data to select data from.

            sel_inds: A numpy array of indices to select.

        Returns:
            A new dictionary with the selected data.
        """
        sel_ts = self.ts_data[key]['ts'][sel_inds]

        vls = self.ts_data[key]['vls']
        if isinstance(vls, np.ndarray):
            sel_vls = self.ts_data[key]['vls'][sel_inds, :]
        else:
            sel_vls = [self.ts_data[key]['vls'][i] for i in sel_inds]
        return {'ts': sel_ts, 'vls': sel_vls}
