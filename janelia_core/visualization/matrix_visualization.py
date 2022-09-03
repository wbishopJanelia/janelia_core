""" Tools for visualizing and comparing matrices.
"""


from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def colorized_tbl(tbl: np.ndarray, dim_0_lbls: list = None, dim_1_lbls: list = None,
                  tbl_fontsize=12, tbl_values_x_offset: float = 0.1, tbl_value_format_str: str = '{:.1e}',
                  show_colorbar = True, cmap: Union[str, matplotlib.colors.Colormap] = None, vmin: float = None,
                  vmax: float = None, ax: plt.Axes = None, label_fontsize=12, xtick_rotation: float = 0.0,
                  ytick_rotation: float = 0.0):
    """ Plots a table of values with colors corresponding to values in each table entry.

    The "tricky" part of using this function is getting the values to have the right font size and be correctly
    positioned in each cell of the table.  Values will always be vertically centered in each cell.  The user can
    specify the font size to make sure all values fit withing a cell, and adjust the x offset of the values to
    make sure the left/right justification looks correct as well.

    Args:
        tbl: The table of values to plot.

        dim_0_lbls: Labels for dimension 0 of the table

        dim_1_lbls: Labels for dimension 1 of the table

        tbl_fontsize: Fontsize in points for the table values

        tbl_values_x_offset: The offset (between 0 and 1) in the x position of table values we plot in each cell.
        A value of 0 corresponds to left justified text.

        tbl_value_format_str: The string to use to format values in the table

        show_colorbar: True if a colorbar should be shown.

        cmap: The colormap to use when mapping colors to values.

        vmin: The lower value that the colormap should satureate at.  If None, will be set from data.

        vmax: The upper value that the colormap should saturate at.  If None, will be set from data.

        ax: Axes to plot the table in.  If None, a new figure and axes will be created.

        label_fontsize: Fontsize in points for axes labels and the optional colorbar

        xtick_rotation: The rotation of x labels

        ytick_rotation: The rotation of y labels
    """
    if ax is None:
        ax = plt.subplot(1,1,1)

    im = ax.imshow(tbl, cmap=cmap, vmin=vmin, vmax=vmax)

    n_rows, n_cols = tbl.shape

    # Show the colorbar if we are suppose to
    if show_colorbar:
        cbar = plt.colorbar(mappable=im)
        cbar.ax.tick_params(labelsize=label_fontsize)

    # Setup the x and y labels
    plt.xticks(range(n_cols), fontsize=label_fontsize, rotation=xtick_rotation)
    plt.yticks(range(n_rows), fontsize=label_fontsize, rotation=ytick_rotation)

    # Provide labels if we are suppose to
    if dim_0_lbls is not None:
        plt.xticks(range(n_cols), dim_0_lbls)

    if dim_1_lbls is not None:
        plt.yticks(range(n_rows), dim_1_lbls)

    # Plot table values - here we center the values vertically in each row
    y_1, y_2 = ax.get_window_extent().get_points()[:, 1]
    points_per_row = (y_2 - y_1)/n_rows
    y_offset_points = tbl_fontsize/2
    y_offset_units = y_offset_points/points_per_row

    for i_0 in range(tbl.shape[0]):
        for i_1 in range(tbl.shape[1]):
            ax.annotate(text=tbl_value_format_str.format(tbl[i_0, i_1]), xy=(i_1-.5 + tbl_values_x_offset,
                        i_0 + y_offset_units), fontsize=tbl_fontsize)


def cmp_n_mats(mats: list, clim: list = None, show_colorbars: bool = False, titles: list = None,
               grid_info: dict = None, cmap: matplotlib.colors.Colormap = None, subplots: list = None) -> list:
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

        cmap: The colormap to use.  Can either be a string or Colormap instance.

        subplots: a list of subplot objects to plot into.  If used, grid_info is ignored

    Returns:
        subplots: List of subplot objects for each subplot showing the matrices in mats.  Subplots are ordered according
        to mats.
    """

    n_mats = len(mats)

    use_existing_subplots = subplots is not None

    # Generate color limits if not provided
    if clim is None:
        min_vl = np.nanmin([np.nanmin(m) for m in mats])
        max_vl = np.nanmax([np.nanmax(m) for m in mats])
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
    if not use_existing_subplots:
        subplots = [None]*n_mats
    img_plots = [None]*n_mats
    for i, m in enumerate(mats):
        if not use_existing_subplots:
            subplot = plt.subplot(grid_spec.new_subplotspec(**grid_info['cell_info'][i]))
        else:
            subplot = subplots[i]

        img_plots[i] = subplot.imshow(m, aspect='auto', cmap=cmap, interpolation='none')

        subplot.set_axis_off()
        if show_colorbars:
            plt.colorbar(img_plots[i], ax=subplot)

        if titles is not None:
            subplot.set_title(titles[i])

        subplots[i] = subplot

    # Make sure all plots are using the same color scaling
    for i_p in img_plots:
        i_p.set_clim(clim)

    return subplots







