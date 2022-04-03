""" Tools for generating images.
"""

import copy
from typing import Callable, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageDraw

from janelia_core.math.basic_functions import list_grid_pts


def alpha_composite(dest: np.ndarray, src: np.ndarray) -> np.ndarray:
    """ Performs alpha compositing with two RGBA images with alpha-premultiplicaiton applied.

    All values should be floating point in the range 0 - 1.

    Standard RGBA values of the form [R_s, G_s, B_s, A_s] can be converted to a
    pre-multiplied RGBA value as [R_s*A_s, G_s*A_s, B_s*A_s, A_s].

    Good notes can be found at: https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Alpha_compositing.html

    Args:
        dest: The bottom image of shape d_x*d_y*4

        src: The top image.  Must be of the same shape as dest_img.

    Returns:
        Nothing.  The dest array will be modified.
    """

    src_alpha = np.expand_dims(src[:, :, 3],2)
    dest[:] = src + dest*(1 - src_alpha)


def generate_2d_fcn_image(f: Callable, dim_0_range: Sequence[float] = None, dim_1_range: Sequence[float] = None,
                          n_pts_per_dim: Sequence[int] = None,
                          vis_dim: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generates an image of a 2d function.

    Args:
        f: The function to visualize. Should accept input of shape [n by 2], where n is an arbitrary number
        of values and output values of length n.

        dim_0_range: The range of dim 0 values to visualize the function over.  If None, [0.0, 1.0] will be used.

        dim_1_range: The range of dim 1 values to visualize the function over.  If None, [0.0, 1.0] will be used.

        n_pts_per_dim: The number of points to sample per dimension.

        vis_dim: If f produces multiple dimension output, this is the dimension to visualize

    Returns:
        im: The image of the function as a 2-d numpy array

        dim_0_pts: The sampled points along dim 0

        dim_1_pts: The sampled points along dim 1
    """

    if dim_0_range is None:
        dim_0_range = [0.0, 1.0]
    if dim_1_range is None:
        dim_1_range = [0.0, 1.0]
    if n_pts_per_dim is None:
        n_pts_per_dim = [1000, 1000]

    pts, dim_pts = list_grid_pts(grid_limits=np.asarray([dim_0_range, dim_1_range]),
                                 n_pts_per_dim=n_pts_per_dim)

    if vis_dim is None:
        vls = f(pts)
    else:
        vls = f(pts)[:, vis_dim].squeeze()

    return vls.reshape(n_pts_per_dim), dim_pts[0], dim_pts[1]


def standard_rgba_to_premultiplied(img: np.ndarray) -> np.ndarray:
    """ Converts between standard and pre-multiplied RGBA representations.

    Args:
        img: Input image in standard RGBA format of shape d_x*d_y*4

    Returns:
        Nothing.  The image is modified in place.
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


def generate_mean_dot_image(image_shape: Sequence[int], dot_ctrs: np.ndarray, dot_vls: np.ndarray,
                                sa_lengths: Sequence[int]) -> np.ndarray:
    """
    Generates a 2-d or 3-d image of the mean of values in ellipsoids.

    Generates an image by associating an ellipsoid with a set of 2 or 3-d locations. Each of these locations
    has an associated value.  A pixel value in the final image is simply the average of all the values associated
    with ellipsoids that contain that pixel.

    Args:
        image_shape: The shape of the image to generate. Must be of length 2 or 3.

        dot_ctrs: The location of each ellipsoid center.  dot_ctrs[i, :] is the location
        for dot i.

        dot_vls: The value to associate with the locations.  dot_vls[i] is associated with the location
        dot_ctrs[i, :]

        sa_lengths: The lengths of the semi-axes of the ellipsoids to generate. The value of sa_lengths[j] is the
        length for dimension j.  All values in sa_lengths must be odd.

    Returns:
        img: The generated image.  Any pixel outside of an ellipsoid will have a value of nan.

    Raises:
        ValueError: If image_shape is not of length 2 or 3.
        ValueError: If all values in sa_lengths are not odd
        ValueError: If any dot centers are outside of the dimensions of the image to be generated
    """

    # Put image shape and sa_lengths into arrays
    image_shape = np.asarray(image_shape).astype('int')
    sa_lengths = np.asarray(sa_lengths).astype('int')

    # Run checks
    if len(image_shape) != 2 and len(image_shape) != 3:
        raise(ValueError('image must be 2-d or 3-d'))

    for l in sa_lengths:
        if l % 2 != 1:
            raise(ValueError('sa_lengths must all be odd'))

    if np.any(dot_ctrs < 0):
        raise(ValueError('One or more dot centers are negative and therefore outside of the image.'))
    if np.any(dot_ctrs > (image_shape-1)):
        raise(ValueError('One or more dot centers exceed image shape values and are therefore outside of the image.'))

    # Generate the basic ellipsoid we convolve throughout the image
    base_e_coord_mats = np.meshgrid(*[np.arange(-l, l+1) for l in sa_lengths], indexing='ij')
    base_e_coord_mats = [(m/l)**2 for m, l in zip(base_e_coord_mats, sa_lengths)]
    base_e_coord_sum = np.zeros(base_e_coord_mats[0].shape)
    for m in base_e_coord_mats:
        base_e_coord_sum = base_e_coord_sum + m
    base_e_im = base_e_coord_sum < 1

    # Generate the empty expanded arrays we need for generating the image
    expanded_im_dims = (image_shape + 2*sa_lengths)
    expanded_im = np.zeros(expanded_im_dims)
    expanded_cnts = np.zeros(expanded_im_dims)

    # Fill the expanded arrays (this is where convolution with the ellipsoid occurs)
    rounded_dot_ctrs = np.round(dot_ctrs).astype('int')
    rounded_shifted_dot_ctrs = rounded_dot_ctrs + sa_lengths
    n_dots = len(dot_vls)
    for d_i in range(n_dots):
        cur_ctr = rounded_shifted_dot_ctrs[d_i, :]
        cur_vl = dot_vls[d_i]
        cur_slice = tuple(slice(c-l, c+l+1) for c, l in zip(cur_ctr, sa_lengths))

        expanded_cnts[cur_slice][base_e_im] += 1
        expanded_im[cur_slice][base_e_im] += cur_vl

    # Produce the final image
    expanded_im = np.divide(expanded_im, expanded_cnts, where=expanded_cnts != 0)
    expanded_im[expanded_cnts == 0] = np.nan

    final_slice = tuple(slice(l, l+d) for d, l in zip(image_shape, sa_lengths))
    return expanded_im[final_slice]


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

    # ==================================================================================================

    # Convert to standard RGBA format
    premultiplied_rgba_to_standard(img)

    # Remove padding from the image
    img = img[offset:-offset, offset:-offset, :]

    return img


def generate_dot_image_3d(image_shape: Sequence[int], dot_ctrs: np.ndarray, dot_vls: np.ndarray,
                          ellipse_shape: Sequence[int]) -> np.ndarray:
    """ Generates a 3-d scalar image of ellipses, given the center location of each ellipse.

    All position/size units are in pixels.

    If two ellipses or more ellipses overlap, the value of the overlapping region is the average of the ellipse values.

    Any pixels in the generated image without an ellipse in them will have the value nan.

    Args:
        image_shape: The shape of the image to generate.

        dot_ctrs: dot_ctrs[i,:] is the center position of the i^th dot in pixels.  Dot centers will be rounded to the
        nearest whole pixel values before generating images of ellipses.

        dot_vls: dot_vls[i] is the value for the i^th ellipse

        ellipse_shape: ellipse_shape[d] gives the width of the generated ellipses in the d^th dimension.  All
        values in ellipse_shape must be odd.

    Returns:
        img: The generated image.

    Raises:
        ValueError: If any of the dot centers are outside of the image.
        ValueError: If any value in ellipse_shape is not odd.
    """

    # Make sure all values in ellipse_shape are odd
    for d in ellipse_shape:
        if d % 2 == 0:
            raise(ValueError('All values in ellipse_shape must be odd.'))

    # Make sure all the centers are in the bounds of the image
    if np.any(dot_ctrs < 0):
        raise(ValueError('All dot centers must be within the bounds of the image'))

    if np.any(dot_ctrs >= image_shape):
        raise(ValueError('All dot centers must be within the bounds of the image'))

    # Generate the mask we will use for the dot
    ellipse_shape = np.asarray(ellipse_shape)
    ellipse_widths = ellipse_shape/2

    x_grid, y_grid, z_grid = np.meshgrid(*[np.arange(d) - np.floor(d/2) for d in ellipse_shape], indexing='ij')
    ellipse_values = ((x_grid**2)/(ellipse_widths[0]**2) + (y_grid**2)/(ellipse_widths[1]**2) +
                      (z_grid**2)/(ellipse_widths[2]**2))

    ellipse_mask = (ellipse_values < 1)

    # Generate the image
    dot_ctrs_rnd  = np.round(dot_ctrs).astype('int')

    # Pad the arrays we need to construct to account for edge effects
    w_sum = np.zeros([image_shape[d_i] + ellipse_shape[d_i] - 1 for d_i in range(3)])
    cnts = np.zeros([image_shape[d_i] + ellipse_shape[d_i] - 1 for d_i in range(3)])
    offsets = np.floor(ellipse_shape/2).astype('int')

    n_dots = dot_ctrs.shape[0]
    for d_i in range(n_dots):
        ctr_i = dot_ctrs_rnd[d_i,:] + offsets
        dot_i = dot_vls[d_i]*ellipse_mask

        dot_slices = tuple([slice(ctr_i[d_i] - offsets[d_i], ctr_i[d_i] + offsets[d_i] + 1) for d_i in range(3)])

        w_sum[dot_slices][ellipse_mask] = w_sum[dot_slices][ellipse_mask] + dot_i[ellipse_mask]
        cnts[dot_slices][ellipse_mask] += 1

    # Generate the final image
    cnts_divide = copy.deepcopy(cnts)
    cnts_divide[cnts == 0] = 1
    img = w_sum/cnts_divide
    img[cnts == 0] = np.nan

    # Remove padding
    img = img[offsets[0]:img.shape[0]-offsets[0], offsets[1]:img.shape[1]-offsets[1],
              offsets[2]:img.shape[2]-offsets[2]]

    return img


def generate_image_from_fcn(f, dim_sampling: Sequence[Sequence]) -> np.ndarray:
    """ Generates a multi-d image from a function.

    The main use for this function is generating images for visualization.

    Args:
        f: The function to plot.

        dim_sampling: Each entry of dim_sampling specifies how to sample a dimension in the
        domain of the fuction.  Each entry is of the form [start, stop, int] where start and
        and stop are the start and stop of the range of values to sample from and int
        is the interval values are sampled from.

    Returns:
        im: The image

        coords: A list of coordinates along each dimension the function was sampled at.
    """

    # Determine coordinates we will sample from along each dimension
    coords = [np.arange(ds[0], ds[1], ds[2]) for ds in dim_sampling]
    n_coords_per_dim = [len(c) for c in coords]

    # Form coordinates of each point we will sample from in a single numpy array
    grid = np.meshgrid(*coords, indexing='ij')
    n_pts = grid[0].size
    flat_grid = np.concatenate([g.reshape([n_pts,1]) for g in grid], axis=1)

    # Evaluate the function
    y = f(flat_grid)

    # Shape y into the appropriate image size
    im = y.reshape(n_coords_per_dim)

    return [im, coords]


def generate_binary_color_rgba_image(a: np.ndarray, neg_clr: Sequence[float] = None,
                                     pos_clr: Sequence[float] = None) -> np.ndarray:
    """ Generates an RGBA image of two colors from a 2-d numpy array.

    Negative values will be one color while positive values will be another.

    The alpha channel will be equal to the absolute value of entries of a, so all values of a must be in [-1, 1].

    The negative color will be assigned to entries strictly less than 0, while the positive channel will be assigned
    to entries greater than or equal to 0.

    Args:
        a: The array to binarize.

        neg_clr: A sequence of floats specifying the RGB values for the negative color.  All value should in [0, 1].
        If None, the negative color will be red.

        pos_clr: A sequence of floats specifying the RGB values for the positive color.  All value should in [0, 1].
        If None, the positive color will be green.

    Returns:
        im: The image of shape [a.shape[0], a.shape[1], 4]

    Raises:
         ValueError: If any entry in a is outside of [-1, 1]
         ValueError: If a is not a 2-d array.
    """

    if np.any(np.abs(a) > 1):
        raise(ValueError('All values in a must be in the range [-1, 1].'))

    if a.ndim != 2:
        raise(ValueError('Input a must be a 2-d array.'))

    im = np.zeros([a.shape[0], a.shape[1], 4])

    if neg_clr is None:
        neg_clr = np.asarray([1, 0, 0])
    else:
        neg_clr = np.asarray(neg_clr)

    if pos_clr is None:
        pos_clr = np.asarray([0, 1, 0])
    else:
        pos_clr = np.asarray(pos_clr)

    neg_vls = a < 0
    im[neg_vls, 0:3] = neg_clr

    pos_vls = a >= 0
    im[pos_vls, 0:3] = pos_clr

    im[:,:,3] = np.abs(a)

    return im


def max_project_pts(dot_positions: np.ndarray, dot_vls: np.ndarray, box_position: np.ndarray,
                    n_divisions: np.ndarray, dot_dim_width: np.ndarray) -> np.ndarray:
    """ Assigns a radius to points in 2-d space to make dots.  Then generates an image of max dot values in a grid.

    The motivation for this function is that there are times when physical objects in space (such as neurons) have
    been represented as finite points.  In these cases, we might desire to visualize how physical properties associated
    with each object are distributed in space.  There are three challenges we face when doing this:

    1) How to visualize a point.  We solve this by adding a radius to the points to make dots.

    2) How to make sense of over-lapping dots.  We do this by taking a "max projection" of the dots - that is
    at each point in space, we keep the value associated with the largest magnitude (so sign doesn't matter).

    3) How to go from continuous space to discrete space, so we can make images.  We solve this by allowing the
    user to specify a grid, defining discrete points in space we want to calculate a value for.

    The final question we need to consider is what to do with points in space that have no dot in them?  In this
    case, we will return an nan value.

    Args:
        dot_positions: The positions of dots, each column is a dimension.  Values must be floating point.

        dot_vls: The values associated with each point.  A 1-d array.

        box_position: The position of a box we define the grid in.  box_position[0,0] is the start of the side for
        dimension 0 and box_position[1,0] is the end of the side for dimension 0.  box_position[:,1] contains the
        start and end of the side for dimension 1.

        n_divisions: The number of divisions to use for the grid.  n_divisions[i] is the number of divisions for
        dimension i.

        dot_dim_width: We allow dots to be ellipses.  dot_dim_width[i] is the width of a dot in dimension i.

    Returns:
        im: The final image, of shape n_divisions.

        inds: Same shape as im.  inds[i,j] is the index of the dot in dot_vls that the value of im[i,j] came from.
        Entries in im with nan values will also have nan values in inds.

    Raises:
        ValueError: If dot positions are not floating point.
        ValueError: If any dot position is outside of the boundaries of the box.
    """

    # Run some checks
    if not dot_positions.dtype == np.dtype('float'):
        raise(ValueError('Dot positions must be floating point numbers.'))

    if np.any(dot_positions[:,0] < box_position[0, 0]) or np.any(dot_positions[:, 0] >= box_position[1,0]):
        raise(ValueError('Dots must be contained within the box.'))

    if np.any(dot_positions[:, 1] < box_position[0, 1]) or np.any(dot_positions[:, 1] >= box_position[1,1]):
        raise(ValueError('Dots must be contained within the box.'))

    box_width = box_position[1, :] - box_position[0, :]

    # Transform points to a standard coordinate system, where lower left of box is at (0,0) and upper right is at (1,1)
    dot_positions = copy.deepcopy(dot_positions)
    dot_positions[:, 0] = dot_positions[:, 0] - box_position[0, 0]
    dot_positions[:, 1] = dot_positions[:, 1] - box_position[0, 1]

    dot_positions[:, 0] = dot_positions[:, 0]/box_width[0]
    dot_positions[:, 1] = dot_positions[:, 1]/box_width[1]

    div_lengths = np.asarray([1/n_divisions[0], 1/n_divisions[1]])

    # Generate a template binary mask of a dot
    dot_n_div = np.asarray([int(np.ceil((dot_dim_width[0]/box_width[0])/div_lengths[0])),
                            int(np.ceil((dot_dim_width[1]/box_width[1])/div_lengths[1]))])

    # Make sure we have an odd number of divisions
    if (dot_n_div[0] % 2) == 0:
        dot_n_div[0] += 1
    if (dot_n_div[1] % 2) == 0:
        dot_n_div[1] += 1

    dot_img = Image.new('1', (dot_n_div[0], dot_n_div[1]))
    dot_drawer = ImageDraw.Draw(dot_img)
    dot_drawer.ellipse((0, 0, dot_n_div[0], dot_n_div[1]), 1)
    dot_img_mask = np.array(dot_img).transpose()
    dot_img_float = dot_img_mask.astype('float')
    dot_img_float[dot_img == 0] = np.nan

    # Create the empty images of max values and indices - initially we add padding
    pad_width = np.floor(dot_n_div/2)
    im = np.zeros([int(n_divisions[0] + 2*pad_width[0]), int(n_divisions[1] + 2*pad_width[1])])
    im[:] = np.nan
    inds = np.zeros_like(im)
    inds[:] = np.nan

    # Now fill in the image
    n_pts = dot_positions.shape[0]
    for p_i in range(n_pts):
        # Calculate center grid square for each dot
        center = np.floor(dot_positions[p_i,:]/div_lengths) + pad_width
        dot_vl = dot_vls[p_i]

        # Calculate the selection coordinates for each grid
        d0_slice = slice(int(center[0] - pad_width[0]), int(center[0] + pad_width[0] + 1))
        d1_slice = slice(int(center[1] - pad_width[1]), int(center[1] + pad_width[1] + 1))

        # Indices where we actually need to do a comparison
        cmp_inds = np.logical_and(dot_img_mask, np.logical_not(np.isnan(im[d0_slice, d1_slice])))

        # See which of compare indices are set to dot values
        keep_cmp_inds = np.greater(np.abs(dot_vl*dot_img_float), np.abs(im[d0_slice, d1_slice]),
                                   where=cmp_inds)

        # See where we don't need to do a comparison
        non_cmp_inds = np.logical_and(dot_img_mask, np.isnan(im[d0_slice, d1_slice]))

        # Get set of all indices we are setting to dot value
        set_inds = np.logical_or(keep_cmp_inds, non_cmp_inds)

        im[d0_slice, d1_slice][set_inds] = dot_vl
        inds[d0_slice, d1_slice][set_inds] = p_i

    # Now remove padding
    d0_im_slice = slice(int(pad_width[0]), int(im.shape[0] - pad_width[0]))
    d1_im_slice = slice(int(pad_width[1]), int(im.shape[1] - pad_width[1]))

    non_padded_im = im[d0_im_slice, d1_im_slice]

    return [non_padded_im, inds]


def rgb_3d_max_project(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    """ Computes 3d-max projection of RGB data.

    The computation is done by converting the image to gray scale, finding max values of the gray scale image,
    and the retaining the rgb information at these max values.

    Args:
        vol: The volume to max project.  Should be of shape [dx, dy, dz, 3], with the last dimension holding RGB values.

        axis: The axis to project along

    Returns:
        proj: The projected image.  The first two dimensions will be the retained dimensions from the projection and
        the last dimension will be of length 3, holding RGB values.

    Raises:
        ValueError: If the last dimension of vol is not of length 3.
    """

    if vol.shape[3] != 3:
        raise(ValueError('vol must be an RGB image.'))

    d_0, d_1, d_2, _ = vol.shape

    # Convert rgb to gray scale using same formula as skimage.color.rgb2gray
    gray_img = .2125*vol[:, :, :, 0] + .7154*vol[:, :, :, 1] + .0721*vol[:, :, :, 2]

    # Find the max along the requested axis
    max_inds = np.argmax(gray_img, axis=axis)

    # Return the projection
    inds_0 = np.arange(d_0).astype('int')
    inds_1 = np.arange(d_1).astype('int')
    inds_2 = np.arange(d_2).astype('int')

    inds_0 = inds_0[:, np.newaxis]
    if axis == 2:
        inds_1 = inds_1[np.newaxis, :]
    else:
        inds_1 = inds_1[:, np.newaxis]
    inds_2 = inds_2[np.newaxis, :]

    inds_sel = [inds_0, inds_1, inds_2, slice(0, 3)]
    inds_sel[axis] = max_inds
    inds_sel = tuple(inds_sel)

    return vol[inds_sel]


def scalar_3d_max_project(vol: np.ndarray, axis: int = 2, abs_vl: bool = False) -> np.ndarray:
    """ Computes 3d max projection of scalar data.

    Args:
        vol: The volume to project.  Should be of shape [dx, dy, dz]

        axis: The dimension to project along

        abs_vl: True if values with the largest absolute value should be returned; if false, then values with the
        largest positive value will be returned.

    Returns:
        proj:  The projected image. Will be 2-d with a shape corresponding to the retained dimensions of vol.

    Raises:
        ValueError: If vol is not a 3d array
    """

    if vol.ndim != 3:
        raise(ValueError('vol must be a 3d array'))

    if abs_vl:
        max_inds = np.argmax(np.abs(vol), axis=axis)
    else:
        max_inds = np.argmax(vol, axis=axis)

    d_0, d_1, d_2 = vol.shape

    # Return the projection
    inds_0 = np.arange(d_0, dtype=int)
    inds_1 = np.arange(d_1, dtype=int)
    inds_2 = np.arange(d_2, dtype=int)

    inds_0 = inds_0[:, np.newaxis]
    if axis == 2:
        inds_1 = inds_1[np.newaxis, :]
    else:
        inds_1 = inds_1[:, np.newaxis]
    inds_2 = inds_2[np.newaxis, :]

    inds_sel = [inds_0, inds_1, inds_2]
    inds_sel[axis] = max_inds
    inds_sel = tuple(inds_sel)

    return vol[inds_sel]


def signed_max_project(volume: np.ndarray, axis: int) -> np.ndarray:
    """ Performs a signed max projection on 3-d data.

    Args:
        volume: The volume to do the max projection on

        axis: The axis to do the max projection along.

    Returns:
        im: The max projection image.  A 2-d array with dimensions inherited from volume.

    Raises:
        ValueError: If volume is not a 3d array
    """

    if volume.ndim != 3:
        raise(ValueError('volume must be a 3-d array.'))

    inds = np.argmax(a=np.abs(volume), axis=axis)

    ret_shape = list(volume.shape)
    ret_shape[axis] = 1
    m_grid = np.meshgrid(*[np.arange(v) for v in ret_shape], indexing='ij')
    m_grid = [m.squeeze() for m in m_grid]
    m_grid[axis] = inds
    m_grid = tuple(m_grid)

    return volume[m_grid]


