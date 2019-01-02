""" Tools for visualizing and comparing matrices.

    William Bishop
    bishopw@hhmi.org
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def cmp_n_mats(mats: list, clim: list = None, show_colorbars: bool = False, titles: list = None,
               grid_info: dict = None) -> list:
    """ Produces a figuring comparing matrices.

    Each matrix will be plotted in a separate axes, and all axes will use the same color scaling.

    Args:
        mats: A list of matrices to compare.

        clim: An list of length 2 of color limits to apply to all images.  If None is provided, will be set to
        [min_v, max_v] where min_v and max_v are the min and max values in all of the matrices

        show_colorbars: True if colorbars should be shown next to each plot

        titles: A list of tiles for each matrix in mats.  If None, no titles will be generated

        grid_info: A dictionary with information about how to layout the matrices in a grid.  It should have the entries:
            grid_spec: The matplotlib.gridspec.GridSpec to use for the grid
            cell_info: A list the same length as mats.  cell_info[i] contains:
                loc: The location for the subplot for the i^th matrix
                rowspan: The row span for the i^th matrix
                colspan: The column span for the i^th matrix

            If grid_info is None, one will be created for showing the matrices next to each other in a row.

    Returns:
        subplots: List of subplot objects for each subplot showing the matrices in mats.  Subplots are ordered according
        to mats.
    """

    n_mats = len(mats)

    # Generate color limits if not provided
    if clim is None:
        min_vl = np.min([np.min(m) for m in mats])
        max_vl = np.max([np.max(m) for m in mats])
        clim = [min_vl, max_vl]

    # Generate default grid info if not provided
    if grid_info is None:
        grid_spec = matplotlib.gridspec.GridSpec(1, n_mats)
        grid_info = {'grid_spec': grid_spec}

        cell_info = [None]*n_mats
        for i in range(n_mats):
            cell_info[i] = {'loc': [0, i], 'rowspan': 1, 'colspan': 1}
        grid_info['cell_info'] = cell_info

    grid_spec = grid_info['grid_spec']

    # Generate plots
    subplots = [None]*n_mats
    img_plots = [None]*n_mats
    for i, m in enumerate(mats):
        subplot = plt.subplot(grid_spec.new_subplotspec(**grid_info['cell_info'][i]))
        img_plots[i] = subplot.imshow(mats[i], aspect='auto')

        subplot.set_axis_off()
        if show_colorbars:
            plt.colorbar(img_plots[i])

        if titles is not None:
            subplot.title = plt.title(titles[i])

        subplots[i] = subplot

    # Make sure all plots are using the same color scaling
    for i_p in img_plots:
        i_p.set_clim(clim)

    return subplots







