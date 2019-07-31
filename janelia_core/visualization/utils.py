""" Basic visualization utilities.

    William Bishop
    bishopw@hhmi.org
"""

import torch
from typing import Sequence

import numpy as np


def alpha_overlay(base_img: np.ndarray, overlay_inds: list, overlay_clrs: np.ndarray,
                  overlay_alpha: np.ndarray) -> np.ndarray:
    """ Takes a base image and overlays another image, performing alpha blending.

    Note: The output of this function will be floating point numbers (not integers).

    Args:
        base_img: The image to overlay things on top of.  Should be of size n_pixels*n_pixels*3,
        where the last dimension contains the RGB values of each pixel in the image.

        overlay_inds: List of length 2.  overlay_inds[i] contains indices of pixels in the i^th
        dimension to overlay

        overlay_clrs: An array of size n_overlay_pixels*3.  overlay_clrs[i, :] gives the RGB
        values of the corresponding pixel to overlay indexed in overlay_inds.

        overlay_alpha: An array of length n_overlay_pixels.  overlay_alpha[i] gives the alpha
        values of the corresponding pixel to overlay indexed in overlay_inds.

    Returns:

        new_image: The new image of size n_pixels*n_pixels*3

    Raises:

        RuntimeError: If inputs do not have expected dimensions.

    Example:

        import numpy as np
        import matplotlib.pyplot as plt

        base_img = 255*np.ones([10, 10, 3], dtype=np.int)

        red_inds = np.indices([4,4])
        red_inds = [np.reshape(i, i.size) for i in red_inds]
        red_clrs = 0*np.ones([16, 3], dtype=np.int)
        red_clrs[:,0] = 255
        red_alpha = 255*.5*np.ones(16, dtype=np.int)

        blue_inds = np.indices([4,4])
        blue_inds = [np.reshape(i, i.size)+2 for i in blue_inds]
        blue_clrs = 0*np.ones([16, 3], dtype=np.int)
        blue_clrs[:,2] = 255
        blue_alpha = 255*.2*np.ones(16, dtype=np.int)

        new_image = alpha_overlay(base_img, red_inds, red_clrs, red_alpha)
        new_image = alpha_overlay(new_image, blue_inds, blue_clrs, blue_alpha)

        plt.imshow(np.ndarray.astype(new_image, np.int))
    """

    if base_img.ndim != 3:
        raise (RuntimeError('base_img must be three dimensional.'))
    if len(overlay_inds) != 2:
        raise (RuntimeError('overlay_inds must be sequence of length 2.'))
    if overlay_clrs.ndim != 2:
        raise (RuntimeError('overlay_clrs must be two dimensional.'))
    if overlay_alpha.ndim != 1:
        raise (RuntimeError('overlay_alpha must be one dimensional.'))

    overlay_alpha = np.expand_dims(overlay_alpha, 1)/255.0

    new_image = base_img
    new_image[overlay_inds[0], overlay_inds[1], :] = \
        overlay_clrs * overlay_alpha + (1 - overlay_alpha) * new_image[overlay_inds[0], overlay_inds[1], :]

    return new_image

