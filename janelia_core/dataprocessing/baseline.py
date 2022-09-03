""" Tools for calculating baseline values for fluorescence data.
"""

import multiprocessing
import os

import numpy as np
from scipy.ndimage.filters import percentile_filter

from janelia_core.stats.histogram import HistogramFilter


def percentile_filter_1d(data: np.ndarray, window_length: int, filter_start: int, write_offset:int,
                         p: float, n_hist_bins: int = 1000) -> np.ndarray:

    """ Performs percentile filtering for a single variable.

    This function uses a moving window to calculate percentiles.  The window is determined by two parameters:
    window_length and filter_start. Window_length is the number of data points in the window.

    The location in the data where filtering is started is determined by filter_start, which gives the
    index of the first value of data that should be in the window.  Negative filter_starts are acceptable
    to indicate a window which begins only partially overlapping the data.

    Finally, write_offset is the offset between the first sample in the window and the sample in the
    output the filtered value should be assigned to.

    Note that windows at the beginning and end of the data may only partially overlap the data.  In these
    cases, only values in the window will be used for calculating percentiles.

    Here are some examples.  To perform standard, acausal filtering with a window of size 101 set:

        window_length = 101

        filter_start = -50

        write_offset = 50

    To perform standard, causal filtering with a window of size 101 set:

        window_length = 101

        filter_start: 0

        write_offset = 100

    Diagram of window placement and parameters::

             Location of first window for filtering:

                        |--------------------|: Window length
                     t=0
                     |
               Data: XXX|XXXXXXXXXXXXXXXXXXXX|XXXX
                        ^                    ^
                        |                    |
                        filter start         window end

     Filtered Data:    |----------|: Write offset
                    YYY|YYYYYYYYYYYYYYYYYYYY|YYYY
                                  ^
                                  |
                      filtered value written here


    Args:

        data: an array of data.

        window_length: The length of samples in a window

        filter_start: The index into the data for the start of the first window.  Negative values indicate
        the first window only partially overlaps the data.

        write_offset: The offset between the first entry in a window and the sample in the output the filtered
        value is written to.

        p: The percentile to use in the baseline calculations.

        n_hist_bins: Percentiles are calculated using a histogram binning technique to improve computational
        efficiency.  This is the number of bins to use in the histogram between the min and max value in the data.

    Returns:

        y: The filtered data.  The same shape as one time point of data. Any points for which a percentile was not
        calculated will be nan.

    Raises:

        RuntimeError: If data with more than 2 dimensions is given as input data

    """

    n_data_pts = len(data)

    min_vl = np.min(data)
    max_vl = np.max(data)

    # Determine which values are in the first window
    first_window_last_ind = np.min([filter_start+window_length-1, n_data_pts-1])
    if first_window_last_ind < 0:
        raise(ValueError('First window falls before the data.'))

    first_window_inds = slice(np.max([0, filter_start]), first_window_last_ind+1)

    # Determine how many times we need to advance the window
    write_ind = filter_start + write_offset
    if write_ind < 0:
        raise(ValueError('First filtered output falls before the data.'))
    if write_ind >= n_data_pts:
        raise (ValueError('First filtered output falls after the data.'))

    n_steps = n_data_pts - write_ind - 1

    # Create the histogram filter and preallocate output
    buffer = HistogramFilter(min_vl, max_vl + .000001, n_hist_bins)

    y = np.empty(n_data_pts)
    y[:] = np.nan

    # Load values in the first window into the buffer
    buffer.reset(data[first_window_inds])
    buffer_bin_edges = buffer.get_bin_starting_edges()

    # Produce first filtered output
    y[write_ind] = buffer_bin_edges[buffer.percentile(p)]

    # Increment window and filter to completion
    prev_filter_start = filter_start
    for step in range(n_steps):
        if prev_filter_start >= 0:
            buffer.remove_value(data[prev_filter_start])

        # Increment start and end indices of the window and the write index
        write_ind += 1
        cur_filter_start = prev_filter_start + 1
        cur_filter_end = cur_filter_start + window_length - 1

        if cur_filter_end < n_data_pts:
            buffer.add_value(data[cur_filter_end])

        # Filter
        y[write_ind] = buffer_bin_edges[buffer.percentile(p)]

        prev_filter_start = cur_filter_start

    return y


def percentile_filter_multi_d(data: np.ndarray, window_length: int, filter_start: int, write_offset, p: float,
                                  mask: np.ndarray = None, n_processes: int = 1) -> np.ndarray:

    """ Calculates baselines independently for each variable in the data across time.

    This function is a wrapper around percentile_filter_1d.  It adds the ability to filter in parellel on
    machines with multiple cores and to filter a restricted set of variabels in a volume, indicated by a mask.

    Args:
        data: an array of data.  First dimension is time.

        window_length, filter_start, write_offset, p: See percentile_filter

        mask: A boolean array the size of one time point of data.  Entries with a value of 1 indicate percentiles for
        that variable should be calculated.  If None, percentiles for all variables will be calculated.

        n_processes: The number of processes to use when calculating baselines.  When calculating baselines for many
        variables on machines with multiple cores, setting this greater than 1 can improve computation time.

    Returns:

        y: The filtered data.  The same shape as one time point of data.  If a mask was used so percentiles were not
        calculated for some variables, the percentiles for these variables will be nan.

    Raises:
        RuntimeError: If data with more than 2 dimensions is given as input data

    """

    n_tm_pts = data.shape[0]
    one_tm_point_shape = data.shape[1:]
    n_dims = len(one_tm_point_shape)

    # Reshape array - we will need this when putting filtered data back into array
    reshape_array = np.ones(n_dims+1, dtype=np.int)
    reshape_array[0] = n_tm_pts

    if mask is None:
        mask = np.ones(one_tm_point_shape, dtype=np.bool)

    # Get coordinates for each point we need to calculate percentiles for
    one_tm_point_inds = [range(s) for s in one_tm_point_shape]
    all_coords = np.mgrid[one_tm_point_inds]
    all_coords = [coords[mask] for coords in all_coords]

    n_vls = len(all_coords[0])
    all_coords = [tuple(slice(all_coords[d][v], all_coords[d][v]+1) for d in range(n_dims))
                  for v in range(n_vls)]
    all_coords = [(slice(None),) + c for c in all_coords]

    # Break up data for processing
    p_data = [{'data': data[c], 'window_length': window_length, 'filter_start': filter_start,
               'write_offset': write_offset, 'p': p} for c in all_coords]

    # Do processing
    if n_processes == 1:
        p_rs = map(_percentile_filter_arg_unpack, p_data)
    else:
        with multiprocessing.Pool(n_processes) as pool:
            p_rs = pool.map(_percentile_filter_arg_unpack, p_data)

    percentiles = np.empty(data.shape)
    percentiles[:] = np.nan

    for c, p_c in zip(all_coords, p_rs):
        percentiles[c] = np.reshape(p_c, reshape_array)

    return percentiles


# Helper function
def _percentile_filter_arg_unpack(kwargs):
    return percentile_filter_1d(**kwargs)


def calculate_causal_percentile_baseline(data: np.ndarray, window_size: int, p: float,
                                         n_hist_bins:int = 1000) -> np.ndarray:
    """ Calculates percentile based baseline in causal manner for one variable.

    The value baseline[i] is calculated as the p-th percentile in the window data[i-window_size+1:i+1].

    Values of baseline[0:window_size] = NAN.

    If data is all a constant value, the non-NAN values in the returned baseline will be that constant value.

    Args:
        data: a 1-d array of data

        window_size: The size of the window to use (units are number of samples)

        p: The percentile to use in the baseline calculations.  Must be in the range [0, 1)

        n_hist_bins: The number of bins to use when calculating histograms for the baseline calculations.

    Returns:
        baseline - an array the same shape as data giving the calculated baseline value at each point.

    """

    n_data_pts = len(data)

    min_vl = np.min(data)
    max_vl = np.max(data)
    baseline = np.full(data.shape, np.nan)
    if max_vl > min_vl:
        buffer = HistogramFilter(min_vl, max_vl + .000001, n_hist_bins)
        buffer.reset(data[0:window_size])
        leading_edges = buffer.get_bin_starting_edges()
        for pt_idx in range(window_size, n_data_pts):
            add_idx = pt_idx
            remove_idx = pt_idx - window_size
            buffer.add_value(data[add_idx])
            buffer.remove_value(data[remove_idx])
            baseline[add_idx] = leading_edges[buffer.percentile(p)]
    else:
        baseline[window_size:-1] = min_vl

    return baseline


def calculate_causal_percentile_baselines(data: np.ndarray, window_size: int, p: float,
                                          n_hist_bins: int=1000,  n_processes=None) -> np.ndarray:
    """ A wrapper function for calculating baseline values for multiple variables in parallel.

    This function wraps around calculate_causal_percentile_baseline, allowing a user to use multiple
    cores to calculate baselines in parallel.

    Args:
        data: An array or size time * (data_shape) with data to calculate baselines for.  Here data_shape is the shape
        of the data at each point in time.

        window_size: See calculate_causal_percentile_baseline()

        p: See calculate_causal_percentile_baseline()

        n_hist_bins: See calculate_causal_percentile_baseline()

        n_processes: The number of processes to use.  If none, this will be set to the number of processors on the
        machine.

    Returns:
        baselines - The baselines for the variables.  Will be the same shape as data.

    """
    if n_processes is None:
        n_processes = int(os.environ['NUMBER_OF_PROCESSORS'])

    # Reshape if needed
    orig_shape = data.shape
    n_time_pts = orig_shape[0]
    n_vars = np.prod(orig_shape[1:])
    reshaped_data = np.reshape(data, [n_time_pts, n_vars], 'C')

    # Calculate baselines
    var_data = [tuple([reshaped_data[:, idx], window_size, p, n_hist_bins]) for idx in range(reshaped_data.shape[1])]
    with multiprocessing.Pool(n_processes) as pool:
        baselines = pool.starmap(calculate_causal_percentile_baseline, var_data)
    baselines = np.asarray(baselines).T

    # Reshape back to original shape
    return np.reshape(baselines, orig_shape, 'C')



