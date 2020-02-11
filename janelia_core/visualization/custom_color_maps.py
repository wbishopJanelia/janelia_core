""" Custom color maps, including color maps over multiple parameters. """

from typing import Sequence

import matplotlib.colors
import numpy as np


class MultiParamCMap():
    """ An object representing a color map over multiple parameters.

    Unlike a typical colormap, which has a single index, these color maps allow indexing over multiple parameters.

    For all colormaps, the user specifies a list of values for each parameter and combinations of these parameter
    values are then mapped to colors.  When using a colormap, values to be assigned colors are rounded to the nearest
    parameter values included in the colormap.

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


def generate_two_param_hsv_map(clr_param_range: np.ndarray, vl_param_range: np.ndarray,
                                      p1_cmap: matplotlib.colors.Colormap, clims: Sequence[float],
                                      vllims: Sequence[float]) -> MultiParamCMap:
    """ Generates a MultiParamCMap for two parameters, which index hue and value of hsv colors.

    Args:
        clr_param_range: The range of values for parameter 0, which indexes into hue, in the form
        (start_vl, stop_vl, step).

        vl_param_range: The range of values for parameter 1, which indexes into value of hsv colors, in the same form
        as clr_param_range.

        p1_cmap: The color map that p1_values will index into. These colors will the be displayed
        for saturation levels of 1.

        clims: The lower and upper parameter values when indexing into p1_cmap.

        vllims: The lower and upper parameter values when indexing into value values.
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


def make_red_blue_green_c_map(n: int = 256) -> matplotlib.colors.LinearSegmentedColormap:
    """ Generates a color map that linearly goes from red at -1 to blue at 0 and then to green at 1.

    Args:
        n: The number of values in the color map

    Returns:
        cmap: The generated color map.
    """

    cdict = {'red': [[0.0, 1.0, 1.0],
                     [0.5, 0.0, 0.0],
                     [1.0, 0.0, 0.0]],
             'green': [[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [1.0, 1.0, 1.0]],
             'blue': [[0.0, 0.0, 0.0],
                      [0.5, 1.0, 1.0],
                      [1.0, 0.0, 0.0]]}

    return matplotlib.colors.LinearSegmentedColormap('red_green', cdict, N=256)


def make_red_green_c_map(n: int = 256) -> matplotlib.colors.LinearSegmentedColormap:
    """ Generates a color map that linearly goes from red at -1, to black at 0 and then to green at 1.

    Args:
        n: The number of values in the color map

    Returns:
        cmap: The generated color map.
    """

    cdict = {'red': [[0.0, 1.0, 1.0],
                     [0.5, 0.0, 0.0],
                     [1.0, 0.0, 0.0]],
             'green': [[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [1.0, 1.0, 1.0]],
             'blue': [[0.0, 0.0, 0.0],
                      [0.5, 0.0, 0.0],
                      [1.0, 0.0, 0.0]]}

    return matplotlib.colors.LinearSegmentedColormap('red_green', cdict, N=256)
