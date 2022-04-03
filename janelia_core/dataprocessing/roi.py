""" Module for working with and representing regions of interest (ROIS) in imaging data. """

import copy

import numpy as np


class ROI():

    def __init__(self, voxel_inds: list, weights: np.ndarray):
        """ Initializes an ROI object.

        Args:
            voxel_inds: This is a tuple of length n_dims.  Each entry contains either (1) the indices of each voxel
            for that dimension or (2) a slice denoting the sides of a hyper rectangle which defines the
            ROI.  In both cases, dimensions are listed in the same order as the image the ROI is for.

            weights: A numpy array of weights for each voxel.  If all voxels have the same weight, this can be a scalar.

        """
        self.voxel_inds = voxel_inds
        self.weights = weights

    @classmethod
    def from_dict(cls, d: dict):
        """ Creates a new ROI object from a dictioary.

        Args:
            d: A dictionary with the keys 'voxel_inds' and 'weights'

        Returns:
            A new ROI object
        """
        return cls(**d)

    @classmethod
    def from_array(cls, arr: np.ndarray, start_inds: np.ndarray = None):
        """ Create an ROI object from a numpy array.

        Args:
            arr: The array represnting the ROI.  Individual values are weights of voxels in the ROI.

            start_inds: If not none, gives the index in a larger image the first index of each dimension
            in arr corresponds to. When start_inds is None, this is equivelent to providing a start_inds of
            all zeros.
        """

        n_dims = len(arr.shape)

        if start_inds is None:
            start_inds = np.zeros(n_dims, dtype=np.int)

        nz_inds = list(np.where(arr))
        shifted_nz_inds = copy.deepcopy(nz_inds)

        for d in range(n_dims):
            shifted_nz_inds[d] = shifted_nz_inds[d] + start_inds[d]

        nz_inds = tuple(nz_inds)
        shifted_nz_inds = tuple(shifted_nz_inds)
        return ROI(shifted_nz_inds, arr[nz_inds])

    def to_dict(self):
        """ Creates a dictionary from a ROI object.

        This is useful for saving the object in a manner which will still allow it to be loaded in the future should
        the class definition of ROI change.

        Returns:
            d: A dictionary with the object data.
        """
        return vars(self)

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

    def center_of_mass(self) -> np.ndarray:
        """ Returns the center of mass of the roi. """

        all_inds = self.list_all_voxel_inds()
        all_w = np.abs(self.list_all_weights())
        total_w = np.sum(all_w)
        return np.asarray([np.sum((i_j*all_w)/total_w) for i_j in all_inds ])

    def list_all_voxel_inds(self) -> list:
        """ Exhaustively lists all voxel coordinates in the roi.

        Returns:
            dim_coords - A tuple listing all voxel indices.
        """

        if isinstance(self.voxel_inds[0], np.ndarray):
           return self.voxel_inds
        else:
            n_dims = len(self.voxel_inds)
            side_lens = [dim_slice.stop - dim_slice.start for dim_slice in self.voxel_inds]
            side_inds = [np.arange(dim_slice.start, dim_slice.stop) for dim_slice in self.voxel_inds]
            voxel_grid = list(np.ndindex(*side_lens))
            dim_coords = [None]*n_dims
            for dim_i in range(n_dims):
                dim_coords[dim_i] = np.asarray([side_inds[dim_i][voxel_ind[dim_i]] for voxel_ind in voxel_grid],
                                               dtype=np.int16)
            return tuple(dim_coords)

    def list_all_weights(self) -> np.ndarray:
        """ Returns weights of each voxel in the roi.

        This function returns an array of weights, even if the weights attribute
        of the ROI object is a scalar (indicating all weights are the same)

        Returns:
            weights: The weights of each voxel in the roi.

            """
        if isinstance(self.weights, int) or len(self.weights) == 1:
            return self.weights*np.ones(self.n_voxels())
        else:
            return self.weights

    def slice_roi(self, plane_idx: int, dim: int = 0, retain_dim=True):
        """ Returns a slice of an ROI.

        Args:
            plane_idx: The index of the plane to slice

            dim: The dimension which defines the plane.

            retain_dim: If true, the voxel_inds of the returned roi will be the same length
            as the original roi.  If false, the entry in voxel_inds for the dimension that was
            sliced along will be removed.

        Returns:
            new_roi: A new roi formed from slicing the roi this function was called on.
        """
        n_dims = len(self.voxel_inds)

        exh_inds = self.list_all_voxel_inds()
        exh_weights = self.list_all_weights()

        keep_inds = np.where(exh_inds[dim] == plane_idx)[0]

        new_voxel_inds = [exh_inds[d][keep_inds] for d in range(n_dims)]
        if retain_dim is False:
            del new_voxel_inds[dim]
        new_voxel_inds = tuple(new_voxel_inds)

        new_weights = exh_weights[keep_inds]

        return ROI(new_voxel_inds, new_weights)

    def intersect_plane(self, plane_idx: int, dim: int = None):
        """ Tests if an ROI intersects a plane.

        Args:
            plane_idx: The index of the plane

            dim: The dimension which defines the plane.  If this is None, dim will be set to 0.

        Returns:
            intersects: True if roi intersects the plane; false if otherwise
        """
        if dim is None:
            dim = 0

        return np.any(self.voxel_inds[dim] == plane_idx)
