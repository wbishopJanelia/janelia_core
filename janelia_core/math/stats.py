""" Tools and classes for working with basic statistics.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
import warnings


class HistogramFilter:
    """ Object for updating a histogram of counts with streaming data.

    This object is useful when needing to calculate percentiles of data over moving windows.

    The user specifies a min and max value of the histograms and a number of bins to include in this range.  The user
    can then add and remove values to the histogram, which will add and remove counts to the appropriate bins in the
    histogram.  If a user tries to add or remove a value falling outside of the range of the histogram, that value will
    be ignored.

    Bins of the histogram have inclusive leading edges and exclusive trailing edges.  For example, if we construct a
    histogram with 2 bins with a min value of 0 and a max value of 2, the bins will have edges [0, .5) and [.5, 1).

    Note: At the moment this function is optimized for speed when adding and removing values but can be subjec to
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
