""" Tools for extracting super voxel ROIS from an imaging datset."""


import numpy as np
import pyspark

from janelia_core.dataprocessing.roi import ROI
from janelia_core.dataprocessing.utils import get_processed_image_data
from janelia_core.dataprocessing.utils import get_reg_image_data


def extract_super_voxels_in_brain(images: list, voxel_size_per_dim: np.ndarray, brain_mask: np.ndarray,
                                  brain_mask_perc: float, image_slice: slice = slice(None, None, None),
                                  t_dict: dict = None, h5_data_group='default',
                                  sc: pyspark.SparkContext=None) -> list:

    """ Extracts super voxel ROIS from imaging data, checking to make sure ROIs are in the brain.

    This function consults a brain mask and only extracts ROIs which overlap a given percentage with the brain mask.

    The value of an ROI at each image is simply the mean of the voxels in the ROI.

    Args:
        images: either (1) a list of images or (2) a list of np.ndarrays, where each array is an image.

        voxel_size_per_dim: The side length individual super voxels in each dimension

        brain_mask: A binary np.ndarray indicating which voxels are in the brain.  A 1 indicates a voxel is
        in the brain.

        brain_mask_perc: The percentage of voxels a super voxel which must overlap with the brain mask in
        order to extract the voxel.

        image_slice: A slice specifying which portion of images to load.  The shape of the image in the slice must
        match the brain mask shape.  If image registration is being used (see t_dict below) the coordinates of this
        slice are for after image registration.

        t_dict: A dictionary with information for performing image registration as images are loaded.  If set to None,
        no image registration will be performed.  t_dict should have two fields:

            transforms: A list of registration transforms to apply to the images as they are being read in.  If none,
            no registration will be applied.

            image_shape: This is the shape of the original images being read in.

        h5_data_group: The data group images are stored in if reading images from .h5 files

        sc: An optional Spark context to speed up computation.

    Returns:
        roi_vls: A np.ndarraay of shape time*n_rois containing the value of each roi at each point in time.

        rois: A list of rois.  Each entry is an ROI object, corresponding to the same column in roi_vls.  The coordinate
        system for the rois is the coordinate system with the slice extracted from the images.

    Raises:
        RuntimeError: If dimensions specified in n_voxels_per_dimension is different than dimensions of images being
        processed.

    """

    do_reg = t_dict is not None

    # Package images with transforms for distributed processing
    if do_reg:
        im0_transform = t_dict['transforms'][0]
        full_image_shape = t_dict['image_shape']
    else:
        im0_transform = None
        full_image_shape = None

    # Get the first full image
    im0 = get_reg_image_data(images[0], image_slice, full_image_shape, im0_transform)
    im_shape = im0.shape
    n_dims = len(im_shape)
    if n_dims != len(voxel_size_per_dim):
        raise(RuntimeError('voxel_size_per_dim must have the same number of dimensions as images being processed.'))

    # Determine voxel placement
    n_voxels_per_dim = [im_shape[i] // voxel_size_per_dim[i] for i in range(n_dims)]
    extra_pixels_per_dim = [im_shape[i] % voxel_size_per_dim[i] for i in range(n_dims)]
    start_pixels_per_dim = [extra_pixs // 2 for extra_pixs in extra_pixels_per_dim]

    super_voxel_per_dim_slices = list()
    for n_voxels, side_length, global_start_ind in zip(n_voxels_per_dim, voxel_size_per_dim, start_pixels_per_dim):
        start_inds = np.arange(global_start_ind, global_start_ind + side_length*n_voxels, side_length)
        dim_slices = [slice(start_ind, start_ind + side_length, 1) for start_ind in start_inds]
        super_voxel_per_dim_slices.append(dim_slices)

    # Generate slices for each supervoxel
    super_voxel_per_dim_slice_inds = list(np.ndindex(*n_voxels_per_dim))
    n_possible_super_voxels = len(super_voxel_per_dim_slice_inds)
    all_slices = [tuple(super_voxel_per_dim_slices[j][super_voxel_per_dim_slice_inds[i][j]] for j in range(n_dims))
                  for i in range(n_possible_super_voxels)]

    # Determine which supervoxels overlap enough with the brain to warrant further processing
    brain_slices = np.where([np.sum(brain_mask[sv_slice])/brain_mask[sv_slice].size >= brain_mask_perc for sv_slice in all_slices])
    brain_slices = [all_slices[j] for j in brain_slices[0]]

    # Extract super voxels
    return extract_super_voxels(images=images, voxel_slices=brain_slices, image_slice=image_slice,
                                t_dict=t_dict, h5_data_group=h5_data_group, sc=sc)


def extract_super_voxels(images: list, voxel_slices: slice, image_slice = slice(None, None, None),
                         t_dict: dict = None, h5_data_group='default', sc: pyspark.SparkContext=None,
                         verbose=True) -> list:

    """ Extracts super voxel ROIS from imaging data.

        The value of the ROI at each image is simply the mean of the voxels in the ROI.

        Args:
            images: either (1) a list of images or (2) a list of np.ndarrays, where each array is an image.

            voxel_slices: A list of slice objects indicating which voxels are in each supervoxel.  Slice coordinates
            correspond to the coordinate system in the extracted slices of images (see below).  So, for example, a
            coordinate of (0,0) corresponds to a corner of image[image_slice] where image is an original image.

            image_slice: A slice specifying which portion of images to load.  The shape of the image in the slice must
            match the brain mask shape.  If image registration is being used (see t_dict below) the coordinates of this
            slice are for after image registration.

            t_dict: A dictionary with information for performing image registration as images are loaded.  If set to None,
            no image registration will be performed.  t_dict should have two fields:

                transforms: A list of registration transforms to apply to the images as they are being read in.  If none,
                no registration will be applied.

                image_shape: This is the shape of the original images being read in.

            h5_data_group: The data group images are stored in if reading images from .h5 files

            sc: An optional Spark context to speed up computation.

            verbose: True if status updates should be printed to screen.

    Returns:
        roi_vls: A np.ndarraay of shape time*n_rois containing the value of each roi at each point in time.

        rois: A list of rois.  Each entry is an ROI object, corresponding to the same column in roi_vls.

        """

    n_images = len(images)
    n_super_voxels = len(voxel_slices)

    if verbose:
        print('Extracting: ' + str(n_super_voxels) + ' super voxels from ' + str(n_images) + ' images.')

    # Extract ROI values
    def extract_rois_from_single_image(image):
        return np.asarray([np.mean(image[sv_slice]) for sv_slice in voxel_slices], dtype=np.float32)

    roi_vls = get_processed_image_data(images=images, func=extract_rois_from_single_image, img_slice=image_slice,
                                       t_dict=t_dict, h5_data_group=h5_data_group, sc=sc)

    roi_vls = np.asarray(roi_vls)

    # Generate roi dictionary
    rois = [None]*n_super_voxels
    for i, sv_slice in enumerate(voxel_slices):
        rois[i] = ROI(voxel_slices[i], 1)

    return [roi_vls, rois]


