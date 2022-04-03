""" Tools for visualizing signals.  """

import matplotlib.pyplot as plt
import numpy as np

from janelia_core.math.basic_functions import find_binary_runs


def plot_segmented_signal(tm_pts: np.ndarray, sig: np.ndarray, ax: plt.Axes = None, delta: float = .6,
                          remove_tm_btw_chunks: bool = True, tm_padding: float = 1.0, color: str = 'k',
                          linewidth: float = 1.0) -> plt.Axes:
    """ Plots a signal that exists at different chunks in time.

    Each chunk will be plotted as a separate trace, and the user can chose to remove time between chunks.

    Args:
        tm_pts: The array of time points the signal is sampled at

        sig: The array of signal values

        ax: Axes to plot into.  If None, a figure with axes will be crated.

        delta: The threshold to use when determining if two sequential time points belong to the same chunk or not

        remove_tm_btw_chunks: True if time periods between chunks should be removed

        tm_padding: The amount of padding (in units of time) to add between chunks

        color: The color to plot the signal in

        linewidth: The linewidth to plot the signal with

    Returns:
        ax: The axis everything was plotted in
    """

    if ax is None:
        f = plt.figure()
        ax = plt.subplot(1, 1, 1)

    # Make sure everything is sorted
    sort_order = np.argsort(tm_pts)
    tm_pts = tm_pts[sort_order]
    sig = sig[sort_order]

    # Find the chunks
    tm_diff = np.diff(tm_pts)
    small_diffs = tm_diff < delta
    runs = find_binary_runs(small_diffs)
    for r_i, run in enumerate(runs):
        runs[r_i] = slice(run.start, run.stop+1)

    # Remove time between chunks if we are suppose to
    if remove_tm_btw_chunks:
        cur_adj = 0
        for run in runs:
            cur_tm_span = tm_pts[run][-1] - tm_pts[run][0]
            tm_pts[run] = (tm_pts[run] - tm_pts[run][0]) + cur_adj
            cur_adj += cur_tm_span + tm_padding

    for run in runs:
        ax.plot(tm_pts[run], sig[run], color=color, linewidth=linewidth)

    return ax
