""" Tools for generating images.

    William Bishop
    bishopw@hhmi.org

"""

from typing import Sequence

import numpy as np
from PIL import Image
from PIL import ImageDraw


def alpha_composite(dest: np.ndarray, src: np.ndarray) -> np.ndarray:
    """ Performs alpha compositing with two RGBA images with alpha-premultiplicaiton applied.

    All values should be floating point in the range 0 - 1.

    Standard RGBA values of the form [R_s, G_s, B_s, A_s] can be converted to a
    pre-multiplied RGBA value as [R_s*A_s, G_s*A_s, B_s*A_s, A_s].

    Good notes can be found at: https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Alpha_compositing.html

    Args:
        dest: The bottom image of shape d_x*d_y*4

        src: The top image.  Must be of the same shape as dest_img.

    Returns: Nothing.  The dest array will be modified.

    """

    src_alpha = np.expand_dims(src[:, :, 3],2)
    dest[:] = src + dest*(1 - src_alpha)


def standard_rgba_to_premultiplied(img: np.ndarray) -> np.ndarray:
    """ Converts between standard and pre-multiplied RGBA representations.

    Args:
        img: Input image in standard RGBA format of shape d_x*d_y*4

    Returns: Nothing.  The image is modified in place.
    """

    alpha = np.expand_dims(img[:, :, 3], 2)
    img[:, :, 0:3] = img[:, :, 0:3]*alpha


def premultiplied_rgba_to_standard(img: np.ndarray) -> np.ndarray:
    """ Converts between pre-multiplied and standard RGBA representations.

    Args:
        img: Input image in pre-multiplied RGBA format of shape d_x*d_y*4

    Returns:
        img: Nothing.  The image is modified in place.
    """

    alpha = np.expand_dims(img[:, :, 3], 2)
    alpha_mask = img[:,:,3] != 0
    img[alpha_mask, 0:3] = img[alpha_mask, 0:3]/alpha[alpha_mask,:]


def generate_dot_image(image_shape: Sequence, dot_ctrs: np.ndarray, dot_clrs: np.ndarray,
                       dot_diameter: int) -> np.ndarray:
    """ Generates an image of dots over a transparent background.

    All position/size units are in pixels.

    Dots are layered according to their order in dot_ctrs with dot_ctrs[0,:] on the bottom.

    Args:
        image_shape: The [width, height] of the image in pixels

        dot_ctrs: dot_ctrs[i,:] gives the position of the center of the i^th dot

        dot_clrs: dot_clrs[i,:] gives the RGBA color of the i^th dot

        dot_diameter: The diameter of the dot to generate in pixel.  Must be an odd number.

    Returns:
        img: The generated image of shape [image_shape[0], image_shape[1], 4] where the last dimension is the RGBA
        value of each pixel.

    """

    img_w = image_shape[0]
    img_h = image_shape[1]

    # Run some checks
    if not isinstance(dot_diameter, int):
        raise(ValueError('dot_diameter must be an integer'))
    if dot_diameter % 2 != 1:
        raise(ValueError('dot diameter must be an odd number'))

    if any((dot_ctrs[:,0] < 0) | (dot_ctrs[:,0] > (img_w - 1))):
        raise(ValueError('All centers must be within image boundaries.'))
    if any((dot_ctrs[:,1] < 0) | (dot_ctrs[:,1] > (img_h - 1))):
        raise(ValueError('All centers must be within image boundaries.'))

    # ==================================================================================================
    # Generate the mask of the template dot we will use
    dot_img = Image.new('1', (dot_diameter, dot_diameter))
    dot_drawer = ImageDraw.Draw(dot_img)
    dot_drawer.ellipse((0, 0, dot_diameter, dot_diameter), 1)
    dot_mask = np.expand_dims(np.array(dot_img),2)

    # ==================================================================================================
    # Place the dots
    dot_ctrs_rnd  = np.round(dot_ctrs).astype('int')

    # Pad the image we will construct to account for edge effects
    img = np.zeros([image_shape[0] + dot_diameter - 1, image_shape[1] + dot_diameter - 1, 4])
    offset = np.floor(dot_diameter / 2).astype('int')

    # Premultiply alpha colors
    dot_clrs_pm = dot_clrs.copy()
    dot_clrs_pm[:, 0:3] = dot_clrs_pm[:, 0:3]*np.expand_dims(dot_clrs[:,3], 1)

    n_dots = dot_ctrs.shape[0]
    for d_i in range(n_dots):

        ctr_i = dot_ctrs_rnd[d_i, :] + offset
        dot_i = dot_mask*dot_clrs_pm[d_i, :]

        dot_slice_0 = slice(ctr_i[0] - offset, ctr_i[0] + offset + 1)
        dot_slice_1 = slice(ctr_i[1] - offset, ctr_i[1] + offset + 1)

        alpha_composite(img[dot_slice_0, dot_slice_1, :], dot_i)
        #print(img[dot_slice_0, dot_slice_1, 0])


    # ==================================================================================================

    # Convert to standard RGBA format
    premultiplied_rgba_to_standard(img)

    # Remove padding from the image
    img = img[offset:-offset, offset:-offset, :]

    return img



