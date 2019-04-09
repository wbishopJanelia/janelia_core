""" Tools for working with data that is saved on disk.

The purpose of these tools is to allow a user to interface with data objects in a standard way
without needing to know if those data objects have been loaded from file or not.  Users can
reference data in these objects, and these objects will handle loading data from disk (and
storing that in memory for future reference).

"""

import pathlib
import os

import h5py
import numpy as np


class NDArrayHandler:
    """ An object for working with numpy.ndarray data. """

    def __init__(self, folder: str, file_name: str, data: np.ndarray = None):
        """ Creates an NDArrayHandler object.

        Args:
            folder: The path to the folder where the file with the data for this object
            is or will be stored.

            file_name: The name of the file which does or will have the data for this object.

            data: The ndarray with the data for this object. Can be left as None.

        """

        self.folder = folder
        self.file_name = file_name
        self._data = data
        self._h5dataset = 'data'

    def __getitem__(self, sl: slice) -> np.ndarray:
        """ Gets data from the object.

        Data will be loaded from disk and saved in memory if this is the first
        time referencing data for this object.

        Args:
            sl: The slice of data to return.

        Returns:
            data: The requested data

        Raises:
            RuntimeError: If NDArrayHandler object has no data to get.

        """

        if self._data is None:
            try:
                self.load_data()
            except FileNotFoundError:
                raise(RuntimeError('NDArrayHandler has no data.'))

        return self._data[sl]

    def __setitem__(self, sl: slice, value: np.ndarray):
        """ Sets data for a slice of an NDArrayHandler object.

        Note: The NDArrayHandler must have some data set already.  To initially set the
        data of the object use the set_data() method.

        Args:
            sl: The slice for data to set

            value: The value to set

        Raises:
            RuntimeError: If the object has no data.

        """

        if self._data is None:
            raise(ValueError('NDArrayobject must have some data before __setitem__ '
                             'is called.  Use .set_data to initially set data. '))

        self._data[sl] = value

    def set_data(self, data: np.ndarray):
        """ Sets the data of the object.

        Args:
            data: the data for the object.
        """
        self._data = data

    def load_data(self):
        """ Loads data from file into memory.

        Raises:
            FileNotFoundError: If the file containing the data is not found.
        """

        # See if file exists and throw an error if it does not
        file_path = pathlib.Path(self.folder) / self.file_name
        if not os.path.isfile(file_path):
            raise FileNotFoundError('File for this NDArrayHandler does not exist: ' + str(file_path))

        # Load data
        with h5py.File(file_path, 'r') as f:
            self._data = f[self._h5dataset][:]

    def save_data(self):
        """ Saves the data in a NDArrayObject to disk. """

        file_path = pathlib.Path(self.folder) / self.file_name
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(self._h5dataset, data=self._data)

    def __getstate__(self):
        """ Returns a new object for pickling and saves the data of this object to file.

        Note: Calling this function will have the side effect of saving the objects data to file.
        This design decision was made because often __getstate__ will be called when pickling a
        NDArrayObject.  In this case, we don't want the data of the object to be saved in data
        file to disk and not in pickled object.  Thus we make sure the data on disk is updated
        before returning an object for pickling which has it's _data attribute set to None.

        """

        if self._data is not None:
            self.save_data()

        state = self.__dict__.copy()
        state['_data'] = None

        return state








