""" Basic visualization utilities.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


def alpha_overlay(base_img: np.ndarray, overlay_inds: np.ndarray, overlay_clrs: np.ndarray):
    """ Takes a base image and overlays another image, performing alpha blending.

    Args:
        base_img: The image to overlay things on top of.  Should be of size n_pixels*n_pixels*4,
        where the last dimension contains the RGBA values of each pixel in the image.

        overlay_inds: A slice object for the pixels in base_img that are to be overlaid.

        overlay_clrs: An array of size n_overlay_pixels*4.  overlay_clrs[i, :] gives the RGBA
        values of the corresponding pixel to overlay indexed in overlay_inds

    """