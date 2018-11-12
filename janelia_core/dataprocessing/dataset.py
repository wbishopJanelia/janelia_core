""" Tools for working with imaging datasets.

    William Bishop
    bishopw@hhmi.org
"""


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
