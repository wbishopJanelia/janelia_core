""" Module for working with and representing rois. """

import numpy as np


class ROI():

    def __init__(self, voxel_inds: list, weights: np.ndarray, vls: np.ndarray=None):
        """ Initializes an ROI object.

        Args:
            voxel_inds: This is tuple of length n_dims.  Each entry contains either (1) the indices of each voxel
            for that dimension or (2) a slice denoting the sides of a hyper rectangle which defines the
            ROI.  In both cases, dimensions are listed in the same order as the image the ROI is for.

            weights: A numpy array of weights for each voxel.  If all voxels have the same weight, this can be a scalar.

        """
        self.voxel_inds = voxel_inds
        self.weights = weights

    def n_voxels(self):
        """ Returns the number of voxels in the roi."""

        if isinstance(self.voxel_inds[0], slice):
            side_lens = [s.stop - s.start for s in self.voxel_inds]
            n_voxels = np.prod(side_lens)
        else:
            n_voxels = len(self.voxel_inds[0])

        return n_voxels

    def bounding_box(self) -> list:
        """ Calculates a bounding box around the ROI.

        Returns:
            A tuple giving slices for each dimension of the bounding box.
        """
        if isinstance(self.voxel_inds[0], slice):
            return self.voxel_inds
        else:
            n_dims = len(self.voxel_inds)
            dim_mins = [np.min(dim_coords) for dim_coords in self.voxel_inds]
            dim_maxs = [np.max(dim_coords) for dim_coords in self.voxel_inds]
            return tuple([slice(dim_mins[i], dim_maxs[i]+1, 1) for i in range(n_dims)])

    def extents(self) -> np.ndarray:
        """ Gets the length of sides of a bounding box holding the roi.

        Returns:
            extents: A np.ndarray of side lengths for each dimension of the bounding box
        """
        bounding_box = self.bounding_box()
        return np.asarray([s.stop - s.start for s in bounding_box])


    def list_all_voxel_inds(self) -> list:
        """ Exhaustively lists all voxel coordinates in the roi.

        Returns:
            dim_coords - A tuple listing all voxel indices.
        """

        if isinstance(self.voxel_inds, tuple):
           return self.voxel_inds
        else:
            n_dims = len(self.voxel_inds)
            side_lens = [dim_slice.stop - dim_slice.start for dim_slice in self.voxel_inds]
            side_inds = [np.arange(dim_slice.start, dim_slice.stop) for dim_slice in self.voxel_inds]
            voxel_grid = np.ndindex(*side_lens)
            dim_coords = [None]*n_dims
            for dim_i in range(n_dims):
                dim_coords[dim_i] = np.asarray([side_inds[dim_i][voxel_ind[dim_i]] for voxel_ind in voxel_grid],
                                               dtype=np.int16)
            return tuple(dim_coords)
