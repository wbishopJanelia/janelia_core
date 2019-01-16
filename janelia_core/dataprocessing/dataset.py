""" Tools for working with imaging datasets.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np

from janelia_core.dataprocessing.roi import ROI


class DataSet:
    """A class for holding a basic data set.

    Args:
        ts_data: A dictionary of time series data.  Each entry in the dictionary is one set of data (with a user
            specified key).  Each set of data is itself stored in a dictionary with two entries.  The fist with
            the key 'ts' is a 1-d numpy.ndarray with timestamps.  The second 'vls' can be:
                1) A list, with one entry per time point
                2) A numpy array of data, with the first dimension corresponding to time
        If data is none the data attribute of the created object will be an empty dictionary.

        metadata: A dictionary of metadata. If meta_data is None, the meta_data attribute of the created object will be
            a empty dictionary.

        **kwargs: Additional keyword arguments that will be added as attributes of the object.
    """

    def __init__(self, ts_data: dict=None, metadata: dict=None, **kwargs):
        if ts_data is None:
            self.ts_data = dict()
        else:
            self.ts_data = ts_data

        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        for k in kwargs:
            setattr(self, k, kwargs[k])

    @classmethod
    def from_dict(cls, d: dict):
        """ Creates a new Dataset object from a dictioary.

        Args:
            d: A dictionary with the keys 'ts_data' and 'metadata'

        Returns:
            A new DataSet object
        """
        return cls(**d)

    def to_dict(self):
        """ Creates a dictionary from a Dataset object.

        This is useful for saving the object in a manner which will still allow it to be loaded in the future should
        the class definition of Dataset change.

        Returns:
            d: A dictionary with the object data.
        """
        return vars(self)

    def has_ts_data(self) -> bool:
        """Returns true if any time series data has non-zero data points.
        """
        ts_keys = self.ts_data.keys()
        has_data = True
        for k in ts_keys:
            has_data = has_data and len(self.ts_data[k]['ts']) > 0
            if not has_data:
                break
        return has_data

    def select_time_range(self, start_time: float, end_time: float):
        """Down selects data in a time range.

        The start and end times are inclusive.

        Args:
            start_time: The start time of the selection.

            end_time: The end time of the selection.

        """

        ts_data_keys = self.ts_data.keys()
        for key in ts_data_keys:
            sel_inds = np.flatnonzero(np.logical_and(self.ts_data[key]['ts'] >= start_time, self.ts_data[key]['ts'] <= end_time))
            self.ts_data[key] = self.select_ts_data(key, sel_inds)

    def select_ts_data(self, key:str, sel_inds: np.ndarray) -> dict:
        """ Selects data by index in a given dictionary of ts_data.

        This function will allow selection to work seamlessly no matter the
        type of 'vls' in the dictionary data is being pulled from.

        Args:
            key: The key of the ts_data to select data from.

            sel_inds: A numpy array of indices to select.

        Returns:
            A new dictionary with the selected data.
        """
        sel_ts = self.ts_data[key]['ts'][sel_inds]

        vls = self.ts_data[key]['vls']
        if isinstance(vls, np.ndarray):
            sel_vls = self.ts_data[key]['vls'][sel_inds, :]
        else:
            sel_vls = [self.ts_data[key]['vls'][i] for i in sel_inds]
        return {'ts': sel_ts, 'vls': sel_vls}

    def ts_paths_to_strings(self):
        """ Looks at time stamps data and converts any path to a string representation.

        This function is useful to call before pickling as it removes problems when trying
        to load paths saved in one operating system (e.g., Windows) on other operating systems
        (e.g., Linux).
        """


class ROIDataset(DataSet):
    """ A dataset object for holding datasets which include roi information.
    """

    def __init__(self, ts_data: dict = None, metadata: dict = None, roi_groups: dict = None, **kwargs):
        """
            Initializes an ROIDataset object.

            ROI data is added to the ts_data dictionary.  A field

        Args:
            ts_data: A dictionary of time series data.  See description in Dataset.__init__()

            metadata: A dictionary of metadata.  See description in Dataset.__init__()

            roi_groups.  A dictionary holding ROI groups.  A "ROI group" is a group of ROIs whose values
                are stored together in entries in ts_data.  One group can have values stored in multiple entries in
                ts_data. Each entry in roi_groups is specified by a key giving the name of the group and a value which
                is a dictioary with the keys:
                    1) rois: A list of ROI objects representing the rois in the group
                    2) ts_labels: A list of ts_data entries with data for this group of rois.
                    3) Optional keys the user may specify. For example, a "type" can be specified.

                If an entry in ts_data holds values for an ROI group, it must hold only values for those ROIs and the order
                of variables in ts_data 'vls' entry must match the order of ROIs in the rois list for the group.

                If roi_groups is none, an empty dictionary will be created.

            **kwargs: Additional keyword arguments that will be added as attributes of the object.

        Raises:

        """
        super().__init__(ts_data, metadata)

        if roi_groups is not None:
            self.roi_groups = roi_groups
        else:
            self.roi_groups = {}

        for k in kwargs:
            setattr(self, k, kwargs[k])

    @classmethod
    def from_dict(cls, d: dict):
        """ Creates a new ROIDataset object from a dictionary.

        Args:
            d: A dictionary with the keys 'ts_data' and 'metadata'

        Returns:
            A new ROIDataset object
        """
        standard_attrs = {'ts_data', 'metadata', 'roi_groups'}

        roi_groups_keys = d['roi_groups'].keys()
        new_roi_groups_dict = dict()
        for k in roi_groups_keys:
            cur_group = d['roi_groups'][k]
            cur_group_keys = set(cur_group.keys())
            cur_group_keys.remove('rois')

            new_group_dict = {k: cur_group[k] for k in cur_group_keys}

            rois_as_objs = [ROI.from_dict(r) for r in cur_group['rois']]
            new_group_dict['rois'] = rois_as_objs

            new_roi_groups_dict[k] = new_group_dict

        nonstandard_attrs = set(d.keys())
        nonstandard_attrs = nonstandard_attrs.difference(standard_attrs)
        nonstandard_dict = {a: d[a] for a in nonstandard_attrs}

        return ROIDataset(d['ts_data'], d['metadata'], new_roi_groups_dict, **nonstandard_dict)

    def to_dict(self) -> dict:
        """ Creates a dictionary from a Dataset object.

        This is useful for saving the object in a manner which will still allow it to be loaded in the future should
        the class definition of Dataset change.

        Returns:
            d: A dictionary with the object data.
        """

        other_attrs = set(vars(self).keys())
        other_attrs.remove('roi_groups')
        save_dict = {a: getattr(self, a) for a in other_attrs}

        roi_groups = self.roi_groups
        roi_group_dict = dict()
        for grp in roi_groups:
            other_group_keys = set(roi_groups[grp].keys())
            other_group_keys.remove('rois')

            new_group_dict = {k: roi_groups[grp][k] for k in other_group_keys}

            rois_as_dict = [r.to_dict() for r in roi_groups[grp]['rois']]
            new_group_dict['rois'] = rois_as_dict

            roi_group_dict[grp] = new_group_dict

        save_dict['roi_groups'] = roi_group_dict

        return save_dict

    def down_select_rois(self, roi_group, roi_inds):
        """ Down select the rois in a dataset.

        ROIs will be removed from dataset.rois and any data for the removed ROIS will
        also be removed from the appropriate ts_data entries.

        Args:
            roi_group: The roi group of the rois to extract

            roi_inds: The indices in dataset.rois to keep.
        """

        new_rois = [self.roi_groups[roi_group]['rois'][i] for i in roi_inds]
        self.roi_groups[roi_group]['rois'] = new_rois

        for label in self.roi_groups[roi_group]['ts_labels']:
            old_vls = self.ts_data[label]['vls']
            new_vls = old_vls[:, roi_inds]
            self.ts_data[label]['vls'] = new_vls

    def extract_rois(self, roi_group, roi_inds, labels):
        """ Extracts rois from a dataset.

        Args:
            roi_inds: The indices of the dataset.rois to extract

            roi_group: The roi group of the rois to extract

            labels: A list of labels of tsdata for the rois to pull out.

        Returns:
            rois: A list of the extracted rois.  Each entry as an ROI object.
            These objects will have fields added holding the ts_data for that ROI.
        """

        if isinstance(labels, str):
            labels = [labels]

        n_rois = len(roi_inds)
        rois = [None]*n_rois
        for i, roi_ind in enumerate(roi_inds):
            dataset_roi = self.roi_groups[roi_group]['rois'][roi_ind]
            roi = ROI(dataset_roi.voxel_inds, dataset_roi.weights)
            for label in labels:
                setattr(roi, label, self.ts_data[label]['vls'][:,roi_ind])
            rois[i] = roi
        return rois

    def form_composite_rois(self, roi_groups:list, roi_weights:list) -> ROI:
        """ Forms composite rois as linear combinations of rois in a dataset.

        Args:
            roi_groups: A list of roi groups to use in forming the composite roi.

            roi_weights: a list of weights.  roi_weights[i] is a np.ndarray of weights for
            the rois in roi_groups[i].  Weights should be listed in the same order as rois in the group
            they are for.

        Returns:
            comp_roi: The composite roi

        """

        # Get maximum possible extent of the composite roi
        min_bounds = [s.start for s in self.roi_groups[roi_groups[0]]['rois'][0].bounding_box()]
        max_bounds = [s.stop for s in self.roi_groups[roi_groups[0]]['rois'][0].bounding_box()]
        n_dims = len(min_bounds)
        for grp_i, grp in enumerate(roi_groups):
            grp_w = roi_weights[grp_i]
            for w_i, roi in enumerate(self.roi_groups[grp]['rois']):
                if grp_w[w_i] != 0:
                    cur_min_bounds = [s.start for s in roi.bounding_box()]
                    cur_max_bounds = [s.stop for s in roi.bounding_box()]
                    min_bounds = [np.min([min_bounds[i], cur_min_bounds[i]]) for i in range(n_dims)]
                    max_bounds = [np.max([max_bounds[i], cur_max_bounds[i]]) for i in range(n_dims)]

        # Create the composite roi in an array
        roi_side_lengths = [max_bounds[i] - min_bounds[i] for i in range(n_dims)]
        comp_roi_array = np.zeros(roi_side_lengths)
        for grp_i, grp in enumerate(roi_groups):
            grp_w = roi_weights[grp_i]
            for w_i, roi in enumerate(self.roi_groups[grp]['rois']):
                if grp_w[w_i] != 0:
                    roi_inds = list(roi.voxel_inds)
                    for d in range(n_dims):
                        roi_inds[d] = roi_inds[d] - min_bounds[d]
                    roi_inds = tuple(roi_inds)
                    comp_roi_array[roi_inds] = comp_roi_array[roi_inds] + grp_w[w_i]*roi.weights

        # Create the composite roi object
        return ROI.from_array(comp_roi_array, min_bounds)

