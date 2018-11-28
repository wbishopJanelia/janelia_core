""" Tools for working with imaging datasets.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np

from janelia_core.cell_extraction.roi import ROI

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
    """

    def __init__(self, ts_data: dict=None, metadata: dict=None):
        if ts_data is None:
            self.ts_data = dict()
        else:
            self.ts_data = ts_data

        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

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

    def select_time_range(self, start_time: float, end_time: float) -> 'DataSet':
        """Selects data in a time range and returns a new dataset Object.

        The start and end times are inclusive.

        Args:
            start_time: The start time of the selection.

            end_time: The end time of the selection.

        Returns:
            New dataset object containing just those time points requested.
        """

        ts_data_keys = self.ts_data.keys()
        sel_ts_data = {}
        for key in ts_data_keys:
            sel_inds = np.flatnonzero(np.logical_and(self.ts_data[key]['ts'] >= start_time, self.ts_data[key]['ts'] <= end_time))
            sel_ts_data[key] = self.select_ts_data(key, sel_inds)
        return DataSet(sel_ts_data, self.metadata)

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


class ROIDataset(DataSet):
    """ A dataset object for holding datasets which include roi information.
    """

    def __init__(self, ts_data: dict = None, metadata: dict = None, roi_groups: dict = None):
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

        Raises:

        """
        super().__init__(ts_data, metadata)

        if roi_groups is not None:
            self.roi_groups = roi_groups
        else:
            self.roi_groups = {}

    def down_select_rois(self, roi_inds):
        """ Down select the rois in a dataset.

        ROIs will be removed from dataset.rois and any data for the removed ROIS will
        also be removed from the appropriate ts_data entries.

        Args:
            roi_inds: The indices in dataset.rois to keep.
        """

        new_rois = [self.rois[i] for i in roi_inds]
        self.rois = new_rois

        for label in self.roi_ts_lbls:
            old_vls = self.ts_data[label]['vls']
            new_vls = old_vls[roi_inds,:]
            self.ts_data[label]['vls'] = new_vls

    def extract_rois(self, roi_inds, labels):
        """ Extracts rois from a dataset.

        Args:
            roi_inds: The indices of the dataset.rois to extract

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
            roi = ROI(self.rois[roi_ind].voxel_inds, self.rois[roi_ind].weights)
            for label in labels:
                setattr(roi, label, self.ts_data[label]['vls'][roi_ind, :])
            rois[i] = roi
        return rois
