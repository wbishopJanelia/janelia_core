""" Tools for extracting ROIS from an imaging datset."""

from typing import Callable, List

import numpy as np
import pyspark

from janelia_core.dataprocessing.roi import ROI
from janelia_core.dataprocessing.utils import get_processed_image_data


def extract_rois(images: list, rois: List[ROI], preprocess_f: Callable = None, t_dict: dict = None,
                 h5_data_group='default', sc: pyspark.SparkContext=None, verbose=True) -> list:

    """ Extracts generate ROIS from imaging data.

        The value of the ROI at each image is simply the mean of the voxels in the ROI.

        Args:
            images: either (1) a list of images or (2) a list of np.ndarrays, where each array is an image.

            rois: A list of rois to extract fluorescence in.

            preprocess_f: An optional function to apply independently to each image after registration but before
            supervoxel extraction.  If None, the function x=f(x) is used.

            t_dict: A dictionary with information for performing image registration as images are loaded.  If set to None,
            no image registration will be performed.  t_dict should have two fields:

                transforms: A list of registration transforms to apply to the images as they are being read in.  If none,
                no registration will be applied.

                image_shape: This is the shape of the original images being read in.

            h5_data_group: The data group images are stored in if reading images from .h5 files

            sc: An optional Spark context to speed up computation.

            verbose: True if status updates should be printed to screen.

    Returns:
        roi_vls: A np.ndarraay of shape time*n_rois containing the value of each roi at each point in time.  The
        order of ROIs corresponds to the rois input.

        """

    n_images = len(images)
    n_rois = len(rois)

    if preprocess_f is None:
        def preprocess_f(x):
            return x

    if verbose:
        print('Extracting: ' + str(n_rois) + ' ROIs from ' + str(n_images) + ' images.')

    # Extract ROI values
    def extract_rois_from_single_image(image):
        image = preprocess_f(image)
        return np.asarray([np.mean(image[roi.voxel_inds]) for roi in rois], dtype=np.float32)

    roi_vls = get_processed_image_data(images=images, func=extract_rois_from_single_image,
                                       t_dict=t_dict, h5_data_group=h5_data_group, sc=sc)

    roi_vls = np.asarray(roi_vls)

    return roi_vls
