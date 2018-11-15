""" Tools for registering imaging data.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


def calc_phase_corr_shift(ref_img: np.ndarray, shifted_img: np.ndarray) -> np.ndarray:
    """ Calculates pixel-wise shift between two images using phase correlation.

    This function works for both 2d and 3d images.

    Args:
        ref_img: The reference image as a numpy array.

        shifted_img: The shifted image as a numpy array.

    Returns: The pixel-wise shift to go from the shifted_img to ref_img.
    """
    pass


def apply_translation(base_img: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """ Applies a shift translation to an image.

    Translation will be performed in Fourier space.

    This function works for both 2d and 3d images.

    Args:
        base_img: The image to shift.

        shift: A vector of shifts in each dimension.

    Returns: The shifted image.
    """
    pass


def get_valid_translated_image_window(shifts: np.ndarray, image_shape: np.ndarray) -> tuple:
    """ Gets a window of an image which is still valid after a shift.

    Returns an window of a shifted image for which pixels in that window were
    shifted versions of pixels in an unshifted image.  This allows us to
    remove pixels from consideration in a shifted which were not based on
    pixels in the original image.

    Multiple shifts can be supplies.  In that case, the window will be valid for
    all shifts.  (Useful when finding valid windows for time series of shifted images).

    Args:
        shifts: The shifts to calculate the window for.  Each column is a shift.

        image_shape: The shape of the image.  Dimensions should be listed here in the
        same order they are listed in shifts.

    Returns: A tuple.  Each entry contains valid indices for a dimension, so that the valid window for an image
    can be recovered as image[t], if t is the returned tuple.

    """
    if shifts.ndim == 1:
        shifts = np.reshape(shifts, [shifts.size, 1])

    shift_margins = np.sign(shifts) * np.ceil(np.abs(shifts))
    shift_maxs = np.ndarray.astype(np.max(shift_margins, 1), 'int')
    shift_mins = np.ndarray.astype(np.min(shift_margins, 1), 'int')

    n_dims = shifts.shape[0]
    return tuple(slice(shift_maxs[i], image_shape[i] - shift_mins[i], 1) for i in range(n_dims))





