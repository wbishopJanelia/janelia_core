""" Custom color maps, including color maps over multiple parameters. """

from typing import Sequence

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


class MultiParamCMap():
    """ An object representing a color map over multiple parameters.

    Unlike a typical colormap, which has a single index, these color maps allow indexing over multiple parameters.

    For all colormaps, the user specifies a list of values for each parameter and combinations of these parameter
    values are then mapped to colors.  When using a colormap, values to be assigned colors are rounded to the nearest
    parameter values included in the colormap.  Saturation is also supported, so that values to be assigned colors outside
    of the range of the values specified for the colormap are assigned colors at the limits of the colormap.

    """

    def __init__(self, param_vl_ranges: Sequence[np.ndarray], clrs: np.ndarray):
        """ Creates a new MultiParamCMap object.

        Args:
            param_vl_ranges: param_vls[i] specifies values for the i^th parameter that specific colors will be assigned
            to and is a tuple of the form (start_vl, stop_vl, step_size), which specifies the values
            start_vl, start_vl + step_size, ... up too all values strictly less than stop_vl.

            clrs: clrs[i_0, i_1, i_2 ... , :] is the color to assign to the combination of param_vls[0][i_0],
            param_vls[1][i_1], param_vls[2][i_2] ...  The number of dimensions in clrs must be equal to the length of
            param_vls plus 1, with the last dimension containing rgb values of colors.

        """

        self.param_vl_ranges = param_vl_ranges
        self.clrs = clrs
        self.n_params = len(param_vl_ranges)
        self.n_param_vls = [len(np.arange(*p_vls)) for p_vls in param_vl_ranges]

    def __getitem__(self, param_vls: Sequence[np.ndarray]) -> np.ndarray:
        """ Returns colors for combinations of parameter values.

        Args:
            param_vls: Values of parameters go generate colors for.  param_vls[i] contains parameter values for the i^th
            parameter.  All entries in param_vls must be numpy arrays of the same shape.

        Returns:
            clrs: The colors at each combination of parameter values.  Will be of shape [*param_vls[0].shape, 3], so
            that clrs[d_0, d_1, d_2, ..., :] is the color for the parameter combination param_vls[0][d_0, d_1, d_2],
            param_vls[1][d_0, d_1, d_2], param_vls[1][d_0, d_1, d_2], ...

        """

        param_0_shape = param_vls[0].shape
        for i in range(1, self.n_params):
            if param_0_shape != param_vls[i].shape:
                raise(IndexError('All param_vls arrays must have the same shape.'))

        inds = [np.round((vls - start)/step) for vls, (start, stop, step) in zip(param_vls, self.param_vl_ranges)]
        inds = [np.minimum(np.maximum(inds_i, 0), n_p_vls_i-1) for inds_i, n_p_vls_i in zip(inds, self.n_param_vls)]
        inds = [inds_i.astype(np.int) for inds_i in inds]
        inds.append(slice(0, 3))
        inds = tuple(inds)
        return self.clrs[inds]

    def to_dict(self):
        """ Returns the attributes of the colormap as a dictionary for serialization."""
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict):
        """ Creates a MultiParamCMap from a dictionary. """
        return cls(param_vl_ranges=d['param_vl_ranges'], clrs=d['clrs'])


def generate_normalized_rgb_cmap(base_map: matplotlib.colors.Colormap, n: int = 1000) -> matplotlib.colors.ListedColormap:
    """ Generates a colormap of RGB values, where each value has a norm of 1.

    Args:
         base_map: The base map to sample.  RGB values of this map will be normalized.

         n: The number of values in the generated map.

    """

    rgb_vls = base_map(np.linspace(0, 1, n))[:, 0:3]
    rgb_norms = np.sqrt(np.sum(rgb_vls**2, axis=1, keepdims=True))
    return matplotlib.colors.ListedColormap(colors=rgb_vls/rgb_norms, name=base_map.name + '_normalized')


def generate_two_param_hsv_map(clr_param_range: Sequence, vl_param_range: Sequence,
                                      p1_cmap: matplotlib.colors.Colormap, clims: Sequence[float],
                                      vllims: Sequence[float]) -> MultiParamCMap:
    """ Generates a MultiParamCMap for two parameters, which index hue and value of hsv colors.

    Args:
        clr_param_range: The range of values for parameter 0, which indexes into hue, in the form
        (start_vl, stop_vl, step). To have a reversed color scale, start_vl should be less than stop_vl and step should
        be a negative value.

        vl_param_range: The range of values for parameter 1, which indexes into value of hsv colors, in the same form
        as clr_param_range.

        p1_cmap: The color map that p1_values will index into. These colors will the be displayed
        for value levels of 1.

        clims: The lower and upper parameter values when indexing into p1_cmap.

        vllims: The lower and upper parameter values when indexing into value values.

    To use a reversed color (value) scale, start_vl should be less than stop_vl and step should be a negative value
    in clr_param_range (vl_param_range) and clims (vllims) should also be flipped so that clims[0] > clims[1].
    """

    clr_param_vls = np.arange(*clr_param_range)
    vl_param_vls = np.arange(*vl_param_range)
    n_vl_vls = len(vl_param_vls)

    # Get base colors
    scaled_clr_vls = np.minimum(np.maximum((clr_param_vls - clims[0])/(clims[1] - clims[0]), 0), 1)
    base_clrs_rgb = p1_cmap(scaled_clr_vls)[:, 0:3]
    base_clrs_hsv = matplotlib.colors.rgb_to_hsv(base_clrs_rgb)

    # Get saturation values
    scaled_vl_param_vls = np.minimum(np.maximum((vl_param_vls - vllims[0]) /
                                                (vllims[1] - vllims[0]), 0), 1)

    # Generate array of color values in hsv format
    clrs_hsv = np.repeat(np.expand_dims(base_clrs_hsv, 1), n_vl_vls, axis=1)
    for v_i in range(n_vl_vls):
        clrs_hsv[:, v_i, 2] = scaled_vl_param_vls[v_i]*clrs_hsv[:, v_i, 2]

    # Convert colors to rgb
    clrs_rgb = matplotlib.colors.hsv_to_rgb(clrs_hsv)

    # Create the MultiParamCMap object
    return MultiParamCMap(param_vl_ranges=[clr_param_range, vl_param_range], clrs=clrs_rgb)


def visualize_two_param_hsv_map(cmap: MultiParamCMap, plot_ax: plt.Axes = None, p0_vls: np.ndarray = None,
                                p1_vls: np.ndarray = None):
    """ Plots a visualization of a two-parameter MultiParamCMap, i.e., the 2-d version of making 1-d colorbar.

    Args:
        cmap: The color map to plot

        plot_ax: The axis to produce the colormap in.  If None, a new figure with axes will be created.

        p0_vls: A list of values to generate the colormap for.

        p1_vls: A list of values to generate the colormap for.

    Raises:
        ValueError: If the colormap is not for two parameters
    """

    if cmap.n_params != 2:
        raise(ValueError('The color map must be for two parameters.'))

    if plot_ax is None:
        plt.figure()
        plot_ax = plt.subplot(1, 1, 1)

    if p0_vls is None:
        p0_vls = np.arange(*cmap.param_vl_ranges[0])
    else:
        p0_vls = np.sort(p0_vls)
    if p1_vls is None:
        p1_vls = np.arange(*cmap.param_vl_ranges[1])
    else:
        p1_vls = np.sort(p1_vls)

    n_p0_smps = len(p0_vls)
    n_p1_smps = len(p1_vls)

    p0_smps = np.repeat(np.expand_dims(p0_vls, 1), n_p1_smps, 1)
    p1_smps = np.repeat(np.expand_dims(p1_vls, 0), n_p0_smps, 0)

    clr_smps = cmap[p0_smps, p1_smps]

    a_ratio = np.abs(p1_vls[-1] - p1_vls[0]) / np.abs(p0_vls[-1] - p0_vls[0])
    plot_ax.imshow(clr_smps, extent=[p1_vls[0], p1_vls[-1], p0_vls[0], p0_vls[-1]], aspect=a_ratio, origin='lower')


def make_red_green_c_map(n: int = 256, inc_transp: bool = False) -> matplotlib.colors.LinearSegmentedColormap:
    """ Generates a color map that linearly goes from red at 0, to black at .5 and then to green at 1.

    Args:
        n: The number of values in the color map

        inc_transp: True if values in the middle of the map (black) should also be transparent.

    Returns:
        cmap: The generated color map.
    """

    if inc_transp:
        middle_alpha = 0.0
    else:
        middle_alpha = 1.0

    return matplotlib.colors.LinearSegmentedColormap.from_list(name='red_to_green', colors=[(0,  [1.0, 0.0, 0.0, 1.0]),
                                                               (.5, [0.0, 0.0, 0.0, middle_alpha]),
                                                               (1.0, [0.0, 1.0, 0.0, 1.0])], N=n)






