""" Tools for extracting super voxel ROIS from an imaging datset."""


import numpy as np
import pyspark

from janelia_core.dataprocessing.roi import ROI
from janelia_core.dataprocessing.utils import get_image_data
from janelia_core.dataprocessing.utils import get_processed_image_data
from janelia_core.math.basic_functions import divide_into_nearly_equal_parts


def extract_super_voxels_in_brain(images: list, n_voxels_per_dimension: np.ndarray, brain_mask: np.ndarray,
                         brain_mask_perc: float, h5_data_group='default', sc: pyspark.SparkContext=None) -> list:
    """ Extracts super voxel ROIS from imaging data, checking to make sure ROIs are in the brain.

    This function consults a brain mask and only extracts ROIs which overlap a given percentage with the brain mask.

    The value of an ROI at each image is simply the mean of the voxels in the ROI.

    Args:
        images - either (1) a list of images or (2) a list of np.ndarrays, where each array is an image.

        n_voxels_per_dimension - A list or numpy array of the number of voxels to extract from each dimension,
        listed in the same order as dimensions in the image.

        brain_mask: A binary np.ndarray the same shape as an image indicating which voxels are in the brain

        brain_mask_perc: The percentage of voxels an a super voxel which must overlap with the brain mask in
        order to extract the voxel.

        h5_data_group: The data group images are stored in if reading images from .h5 files

        sc: An optional Spark context to speed up computation.

    Returns:
        roi_vls: A np.ndarraay of shape time*n_rois containing the value of each roi at each point in time.

        rois: A list of rois.  Each entry is an ROI object, corresponding to the same column in roi_vls.

    Raises:
        RuntimeError: If dimensions specified in n_voxels_per_dimension is different than dimensions of images being
        processed.

    """

    # Get the first full image
    im0 = get_image_data(images[0], h5_data_group=h5_data_group)
    im_shape = im0.shape
    n_dims = len(im_shape)
    if n_dims != len(n_voxels_per_dimension):
        raise(RuntimeError('n_voxels_per_dimension must have the same number of dimensions as images being processed.'))

    # Determine voxel placement
    super_voxel_lengths = [divide_into_nearly_equal_parts(im_shape[i], n_voxels_per_dimension[i]) for i in range(len(im_shape))]

    super_voxel_per_dim_slices = list()
    for dim_sv_lengths in super_voxel_lengths:
        start_inds = np.cumsum(np.concatenate((np.asarray([0]), dim_sv_lengths)))
        dim_slices = [slice(start_inds[i], start_inds[i] + dim_sv_lengths[i], 1) for i in range(len(dim_sv_lengths))]
        super_voxel_per_dim_slices.append(dim_slices)

    # Generate slices for each supervoxel
    super_voxel_per_dim_slice_inds = list(np.ndindex(*n_voxels_per_dimension))
    n_possible_super_voxels = len(super_voxel_per_dim_slice_inds)
    all_slices = [tuple(super_voxel_per_dim_slices[j][super_voxel_per_dim_slice_inds[i][j]] for j in range(n_dims))
                  for i in range(n_possible_super_voxels)]

    # Determine which supervoxels overlap enough with the brain to warrant further processing
    brain_slices = np.where([np.sum(brain_mask[sv_slice])/brain_mask[sv_slice].size >= brain_mask_perc for sv_slice in all_slices])
    brain_slices = [all_slices[j] for j in brain_slices[0]]

    # Extract super voxels
    return extract_super_voxels(images, brain_slices, h5_data_group, sc)


def extract_super_voxels(images: list, voxel_slices: slice, h5_data_group='default',
                        sc: pyspark.SparkContext=None, verbose=True) -> list:
    """ Extracts super voxel ROIS from imaging data.

        The value of the ROI at each image is simply the mean of the voxels in the ROI.

        Args:
            images: either (1) a list of images or (2) a list of np.ndarrays, where each array is an image.

            voxel_slices: A list of slice objects indicating which voxels are in each supervoxel

            h5_data_group: The data group images are stored in if reading images from .h5 files

            sc: An optional Spark context to speed up computation.

            verbose: True if status updates should be printed to screen.

    Returns:
        roi_vls: A np.ndarraay of shape time*n_rois containing the value of each roi at each point in time.

        rois: A list of rois.  Each entry is an ROI object, corresponding to the same column in roi_vls.

        """

    n_super_voxels = len(voxel_slices)

    if verbose:
        print('Extracting: ' + str(n_super_voxels) + ' super voxels.')

    # Extract ROI values
    def extract_rois_from_single_image(image):
        return np.asarray([np.mean(image[sv_slice]) for sv_slice in voxel_slices], dtype=np.float32)
    roi_vls = get_processed_image_data(images, extract_rois_from_single_image, h5_data_group, sc)
    roi_vls = np.asarray(roi_vls)

    # Generate roi dictionary
    rois = [None]*n_super_voxels
    for i, sv_slice in enumerate(voxel_slices):
        rois[i] = ROI(voxel_slices[i], 1)

    return [roi_vls, rois]


