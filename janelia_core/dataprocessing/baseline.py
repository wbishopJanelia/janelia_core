""" Tools for calculating baseline values for fluorescence data.

    William Bishop
    bishopw@hhmi.org
"""

import multiprocessing
import os
from typing import Sequence

import numpy as np

from janelia_core.math.stats import HistogramFilter


def calculate_causal_percentile_baseline(data, window_size: int, p: float,
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

    This funcion wraps around calculate_causal_percentile_baseline, allowing a user to use multiple
    cores to calculate baselines in parallel.

    Args:
        data: An array or size time * (data_shape) with data to calculate baselines for.  Here data_shape
        is the shape of the data at each point in time.

        window_size: See calculate_causal_percentile_baseline()

        p: See calculate_causal_percentile_baseline()

        n_hist_bins: See calculate_causal_percentile_baseline()

        n_processes: The number of processes to use.  If none, this will be set to the number of processors
        on the machine.

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



