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
            the key 'ts' is a 1-d numpy.ndarray with timestamps.  The second 'vls' is a list of or numpy array of data
            for each point in ts. If data is none the data attribute of the created object will be an empty dictionary.

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

    def __init__(self, ts_data: dict = None, metadata: dict = None, rois: list = None,
                 roi_ts: np.ndarray = None, roi_data_labels: list = None, roi_data: list = None):
        """
            Initializes an ROIDataset object.

            ROI data is added to the ts_data dictionary.  A field

        Args:
            ts_data: A dictionary of time series data.  Each entry in the dictionary is one set of data (with a user
            specified key).  Each set of data is itself stored in a dictionary with two entries.  The fist with
            the key 'ts' is a 1-d numpy.ndarray with timestamps.  The second 'vls' is a list of or numpy array of data
            for each point in ts. If data is none the data attribute of the created object will be an empty dictionary.

            metadata: A dictionary of metadata. If meta_data is None, the meta_data attribute of the created object will be
            a empty dictionary.

            rois: A list with ROI objects. If rois is none, an empty list will be created.

            rois_ts: A ndarray of time stamps for roi data

            roi_data_labels: A list of labels for the types of data provided in roi_data.

            roi_data: A list of numpy arrays with roi data.  Each entry is a ndarray of size n_rois * t,
            where t is the number of time stamps in rois_ts.  roi_data[i,j] contains the value of rois[i]
            at time rois_ts[j].  If this is none, no ROI data will be added.
        """
        super().__init__(ts_data, metadata)

        if rois is not None:
            self.rois = rois
        else:
            self.rois = []

        if roi_data is not None:
            self.roi_ts_lbls = roi_data_labels
            for i, roi_data_type in enumerate(roi_data_labels):
                self.ts_data[roi_data_type] = {'ts': roi_ts, 'vls': roi_data[i]}

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
