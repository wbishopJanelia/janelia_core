""" Tools and classes for working with basic statistics.

    William Bishop
    bishopw@hhmi.org
"""

import multiprocessing
import os
from sortedcontainers import SortedList
from typing import Sequence
import warnings

import numpy as np


class HistogramFilter:
    """ Object for updating a histogram of counts with streaming data.

    This object is useful when needing to calculate percentiles of data over moving windows.

    The user specifies a min and max value of the histograms and a number of bins to include in this range.  The user
    can then add and remove values to the histogram, which will add and remove counts to the appropriate bins in the
    histogram.  If a user tries to add or remove a value falling outside of the range of the histogram, that value will
    be ignored.

    Bins of the histogram have inclusive leading edges and exclusive trailing edges.  For example, if we construct a
    histogram with 2 bins with a min value of 0 and a max value of 1, the bins will have edges [0, .5) and [.5, 1).

    Note: At the moment this function is optimized for speed when adding and removing values but can be subject to
    floating point errors.  If more exact results are required, used other means of calculating histograms.

    """

    def __init__(self, min=0, max=0, n_bins=1000):
        """ Creates a HistogramFilter.

        Args:
            min,max: Set the range of values we want to calculate histograms values over.  Values outside of this
            range will be ignored when adding or removing values from the buffer.

            n_bins: The number of bins in the histogram.

        """
        self.min = min
        self.max = max
        self.n_bins = n_bins

        # Setup buffer
        self.buffer = np.zeros(n_bins)

    def reset(self, data: np.ndarray = None):
        """ Resets the buffer.

        If no data is provided, all counts in the buffer are set to 0.  If data is provided,
        the buffer is filled with counts of the values in the provided data.

        Args:
            data: If provided, counts in the buffer are reset based on this data.
        """
        self.buffer = np.zeros(self.n_bins)
        if data is not None:
            for vl in data:
                self.add_value(vl)

    def add_value(self, vl):
        """ Adds a value to the buffer."""
        bin_idx = self._get_bin(vl)
        if bin_idx is not None:
            self.buffer[bin_idx] += 1

    def remove_value(self, vl):
        """ Removes a value from the buffer."""
        bin_idx = self._get_bin(vl)
        if bin_idx is not None:
            self.buffer[bin_idx] -= 1

    def percentile(self, percentile: float) -> int:
        """ Returns the bin containing a given percentile.

        By "bin containing given percentile" this function returns the smallest index of a bin
        for which the counts in that bin exceed the requested percentile.

        Args:
            percentile: The requested percentile

        Returns:
            bin_idx: The bin containing the requested percentile.

        Raises:
            RuntimeError: If percentile is outside of the range [0, 1).
        """

        if percentile < 0 or percentile >= 1:
            raise(RuntimeError('Percentile must be in the range [0, 1).'))

        cum_counts = np.cumsum(self.buffer)
        bin_percentiles = cum_counts/cum_counts[-1]
        return np.argmax(bin_percentiles >= percentile)

    def get_bin_starting_edges(self) -> np.ndarray:
        """ Returns an array of starting edges for each bin in the buffer.
        """
        bin_size = (self.max - self.min)/self.n_bins
        return np.arange(self.min, self.max, bin_size)

    def _get_bin(self, vl) -> int:
        """ Returns the bin of a histogram a value falls into.

        If the value falls outside of the range of the histogram (defined by the min and max values of the
        HistogramBuffer object) a warning will be printed and None will be returned.

        Note: This function is not the most robust with respect to floating point errors and could probably
        be improved in this regard.

        Args:
            vl: The value the bin should be returned for

        Returns:
            bin: The index of self.buffer for the bin the value falls in.
        """

        norm_value = (vl - self.min) / (self.max - self.min)
        bin = int(np.floor(norm_value*self.n_bins))

        if bin < 0:
            warnings.warn(RuntimeWarning('The value ' + str(vl) +
                                         ' is less than the min value for the histogram buffer (' + str(self.min) + ')'))
            return None

        if bin >= self.n_bins:
            warnings.warn(RuntimeWarning('The value ' + str(vl) +
                                         ' is greaten than or equal to the max value for the histogram buffer (' + str(self.max) + ')'))
            return None

        return bin


class NDPercentileFilter:
    """ An object for applying a percentile filter to streaming n-dimensional data.

    Percentile filters are applied across time separately to each point in n-dimensional data.

    This object is useful when calculating percentiles in a streaming manner. The idea is that data can be added to and
    removed from buffers.  Buffers are not of fixed length and efficient manners of updating sorted values in the buffers
    are used.  At any time, a user can request the p-th percentiles of data in the buffers, which amounts to returning
    data from a fixed index in each buffer and is therefore efficient.

    """

    def __init__(self, data_shape: Sequence[int], mask: np.ndarray = None, n_processes = None):
        """ Creates a PercentileFilter object.

        Args:
            data_shape: The shape of the data at each point.  E.g., if calculating the percentiles of pixel values
            across time of 2-d images, data_shape would be the height and width of an individual images.

            mask: A binary array of shape data_shape.  Each entry indicates if percentiles should be calculated
            for that entry in the data, with 1 indicating a value should be calculated.  If None, percentiles
            will be calculated for all entries.

            n_processes: The number of processes to use.  If None, this will be set to the number of
            processors on the machine.

        """

        n_dims = len(data_shape)
        self.data_shape = data_shape

        if mask is None:
            mask = np.ones(data_shape, dtype=np.bool)
        self.mask = mask

        # Setup buffer
        n_vls = np.sum(self.mask)  # How many values there are in each data point
        self.n_vls = n_vls
        self.sorted_vl_buffer = [SortedList() for _ in range(n_vls)]  # Buffer for storing sorted values

        # Now we create some variables for helping us go from a flat representation of our data to a ndarray (mgrid) and
        # vice versa (point_coords)
        coord_list = [slice(0, d) for d in data_shape]
        mgrid = np.mgrid[coord_list]
        self.mgrid_flat = tuple(a[self.mask] for a in mgrid)
        self.point_coords = [tuple(self.mgrid_flat[d][i] for d in range(n_dims)) for i in range(n_vls)]

        # Determine how many processes to use
        if n_processes is None:
            n_processes = int(os.environ['NUMBER_OF_PROCESSORS'])
        self.n_processes = n_processes

        # Create the pool to use
        self.pool = multiprocessing.Pool(n_processes)

    def add_data(self, data: np.ndarray):
        """ Adds a point of data to a percentile filter.

        Args:
            data: The point of data to add
        """

        buffers_with_data = [tuple([self.sorted_vl_buffer[i], data[self.point_coords[i]]])
                             for i in range(self.n_vls)]

        self.sorted_vl_buffer = self.pool.starmap(_add_data_to_sorted_list, buffers_with_data)


    def remove_data(self, data: np.ndarray):
        """ Removes a point of data from a percentile filter.

        Args:
            data: The point of data to remove.
        """

        for i in range(self.n_vls):
            self.sorted_vl_buffer[i].discard(data[self.point_coords[i]])

    def retrieve_percentile(self, p: float) -> np.ndarray:
        """ Returns percentile values based on buffered data.

        Args:
            p: The requested percentile in the range [0, 1]

        Returns: The percentile for each point in the data.  Will be the same shape as the data.

        Raises:
            RuntimeError: If no data has been added to buffers yet.
        """
        # Determine where in the sorted lists the percentile we need to pull will be found
        buffer_l = len(self.sorted_vl_buffer[0])

        if buffer_l == 0:
            raise(RuntimeError('No data in buffers.'))

        index = np.ceil(p*(buffer_l-1)).astype('int')

        # Pull the value from each list
        percentiles_list = [self.sorted_vl_buffer[i][index] for i in range(self.n_vls)]

        # Return percentiles in a numpy array the same shape as our data
        percentiles = np.zeros(self.data_shape)
        percentiles[self.mgrid_flat] = percentiles_list
        return percentiles


# Helper functions go here
def _add_data_to_sorted_list(l, d):
    l.add(d)
    return l

