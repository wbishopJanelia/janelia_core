""" Tools for representing and working with datasets. """

from typing import Sequence

import numpy as np
import torch
import torch.utils.data


class TimeSeriesBatch(torch.utils.data.Dataset):
    """ An object to represent a batch of samples from time series data.

    Each sample in the batch is a pair consisting of data at time t-1 and data at time t.
    """

    def __init__(self, data: Sequence[torch.Tensor], i_x: torch.Tensor, i_y: torch.Tensor, i_orig: torch.Tensor):
        """

        Args:
            data: data[i] contains data for the i^th group of variables.  Dimension 0 indexes individual points.

            i_x: Indicates which points in data correspond to the t-1 data points in each sample.

            i_y: Indicates which points in data correspond to the t data points in each sample.

            i_orig: Indicates the original indices in the larger TimeSeriesData object the data points came from.
                    (Keeping track of this is helpful when needing to merge and concatenate batches later.)

        """
        super().__init__()
        self.data = data
        self.i_x = i_x
        self.i_y = i_y
        self.i_orig = i_orig

    def pin_memory(self):
        """ Puts this object's data into pinned memory. """

        for i in range(len(self.data)):
            self.data[i] = self.data[i].pin_memory()

        self.i_x = self.i_x.pin_memory()
        self.i_y = self.i_y.pin_memory()
        self.i_orig = self.i_orig.pin_memory()

        return self

    def to(self, device:torch.device, move_i_orig: bool = False, non_blocking:bool =False):
        """ Moves data to the specified device.

        Args:
            device: The device to move the data to.

            move_i_orig: If true, i_orig will be moved to the device.  If false, only data, i_x and i_y will be
            moved to the device.

            non_blocking: If the .to() command should be passed the non_blocking option.
        """

        self.data = [t.to(device, non_blocking=non_blocking) for t in self.data]
        self.i_x = self.i_x.to(device, non_blocking=non_blocking)
        self.i_y = self.i_y.to(device, non_blocking=non_blocking)
        if move_i_orig:
            self.i_orig = self.i_orig.to(device, non_blocking=non_blocking)

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.i_x)

    def __getitem__(self, index):
        """ Returns requested samples from the dataset.

        Args:
            index: Integer index, slice or numpy array indicating requested samples.
            The index is specifically into the i_x and i_y attributes.

        Returns:
            smps: The requested samples.
        """

        # Get unique list of i_orig values for the samples we are to keep
        keep_i_orig_x = self.i_orig[self.i_x[index]].cpu().numpy()
        keep_i_orig_y = self.i_orig[self.i_y[index]].cpu().numpy()

        if keep_i_orig_x.ndim == 0:
            keep_i_orig_x = np.expand_dims(keep_i_orig_x, 0)

        if keep_i_orig_y.ndim == 0:
            keep_i_orig_y = np.expand_dims(keep_i_orig_y, 0)

        keep_i_orig = np.unique(np.concatenate((keep_i_orig_x, keep_i_orig_y)))

        # Find out where the i_orig values that we are keeping come from in the original data array
        pull_inds = torch.tensor(tuple(np.nonzero(self.i_orig == ind)[0] for ind in keep_i_orig)).long()

        # Pull the values for the new data and i_orig
        pulled_data = [t[pull_inds] for t in self.data]
        pulled_i_orig = self.i_orig[pull_inds]

        # Form the new i_x and i_y
        pulled_i_x = torch.tensor(tuple(np.nonzero(pulled_i_orig == i)[0] for i in keep_i_orig_x)).long()
        pulled_i_y = torch.tensor(tuple(np.nonzero(pulled_i_orig == i)[0] for i in keep_i_orig_y)).long()

        return TimeSeriesBatch(data=pulled_data, i_x=pulled_i_x, i_y=pulled_i_y, i_orig=pulled_i_orig)

    def efficient_get_item(self, index):
        """ A computational (but not memory) efficient way of indexing a TimeSeriesBatch object.

        A new TimeSeriesBatch object will be returned with unmodified data and i_orig attributes.  The i_x
        and i_y attributes will be updated to index the data and i_orig attributes.  By not discarding
        any unused points in the data attribute, indexing can be done very quickly.

        This function can be useful when seeking to index an existing TimeSeriesBatch object when we will not
        need to copy the result to another place in memory (e.g., sending to a GPU when the indexed TimeSeriesBatch
        object is on CPU).

        Args:
            index: Integer index, slice or numpy array indicating requested samples.
            The index is specifically into the i_x and i_y attributes.

        Returns:
            smps: The requested samples in a TimeSeriesBatch object as described above.
        """
        if isinstance(index, int):
            index = torch.Tensor([index]).long()
        return TimeSeriesBatch(data=self.data, i_x=self.i_x[index], i_y=self.i_y[index], i_orig=self.i_orig)


class TimeSeriesDataset(torch.utils.data.Dataset):
    """ Extends torch's Dataset object specifically for time series data.

    This object is specifically designed for scenarios when we want to predict points at time t + 1 from points at
    time t.

    The user supplies the time series data for different groups of variables.  He or she can then request samples, which
    will return the time series data for the samples. Each sampled data point is a pair corresponding to data at
    time t and the data a time t + 1.  Therefore, a dataset with T time points will contain T-1 samples.

    The requested samples will be returned in a memory efficient manner to avoid representing duplicate points
    twice.

    """

    def __init__(self, ts_data: Sequence[torch.Tensor]):
        """ Creates a TimeSeriesDataset object.

        Args:
            ts_data: Time series data. ts_data[i] is time series data for the i^th group of variables.  Time is
            represented along dimension 0.  All ts_data tensors must have the same number of time points.

        Raises:
            ValueError: If not all ts_data tensors have the same number of time points.
        """
        super().__init__()

        n_ts = len(ts_data)

        if np.any([ts_data[0].shape[0] != ts_data[i].shape[0] for i in range(n_ts)]):
            raise(ValueError('All tensors in ts_data must have same number of time points.'))

        self.ts_data = ts_data

    def __len__(self):
        """ Returns the number of samples in the dataset.

        Note that number of samples in the dataset is 1 less than the number of time points, as samples
        consist of pairs of data points at time t-1 and time t.

        Returns:
            The number of samples in the dataset.
        """
        return self.ts_data[0].shape[0] - 1

    def __getitem__(self, index) -> TimeSeriesBatch:
        """ Returns requested samples from the dataset.

        A single sample consists of a pair of data points at time t-1 and at time t.

        Args:
            index: Integer index or slice indicating requested samples.

        Returns:
            smps: The requested samples.

        """

        if isinstance(index, int):
            index = np.asarray([index])

        # Determine indices into the dataset for all points in the pairs
        all_indices = np.arange(0, len(self))
        x_indices = all_indices[index] # Indices for points in each sample at time t
        y_indices = x_indices + 1 # Indices for points in each sample at time t + 1
        union_indices = np.asarray(list(set(x_indices).union(set(y_indices))))

        # Ensure indices are sorted
        union_indices = np.sort(union_indices)

        # Form x_idx & y_idx
        x_idx = np.asarray([np.nonzero(union_indices == x_i)[0][0] for x_i in x_indices])
        y_idx = np.asarray([np.nonzero(union_indices == y_i)[0][0] for y_i in y_indices])
        x_idx = torch.Tensor(x_idx).long()
        y_idx = torch.Tensor(y_idx).long()

        # Form output
        union_indices = torch.Tensor(union_indices).long()

        return TimeSeriesBatch(data=[tensor[union_indices] for tensor in self.ts_data],
                               i_x=x_idx,
                               i_y=y_idx,
                               i_orig=union_indices)


def cat_time_series_batches(batches: Sequence[TimeSeriesBatch]) -> TimeSeriesBatch:
    """ Concatenates multiple TimeSeriesBatches into one object in a memory efficient manner.
    
    The i_x and i_y fields of the new TimeSeriesBatch object will effectively be concatenations of the
    i_x and i_y fields from the original, individual TimeSeriesBatch objects, though the values of i_x and
    i_y may be changed to appropriately index into the data tensors of the returned TimeSeriesBatch object.
    
    Args:
        batch: A list of TimeSeriesBatch objects to concatenate.
        
    Returns:
        conc_batch: The TimeSeriesBatch object representing the concatenated result.
    """

    # ===============================================================
    # Merge the data of all batches
    n_grps = len(batches[0].data)

    # Construct a lookup table
    lut = torch.cat(tuple(torch.stack(tuple([b.i_orig, # Original indices each data point came from
                                  b_i*torch.ones(len(b.i_orig)).long(), # Batch each data point came from
                                  torch.arange(len(b.i_orig)) # Position in each batch of each sample
                                      ]), dim=1) for b_i, b in enumerate(batches)), dim=0)

    # Determine what the unique sample time points are and make sure they are sorted
    cat_i_orig = torch.unique(lut[:, 0], sorted=True)
    lut_rows = torch.Tensor(tuple(torch.nonzero(lut[:,0] == u_i)[0] for u_i in cat_i_orig)).long()

    # Now form the merged data for each group
    cat_data = [None]*n_grps
    for g_i in range(n_grps):
        cat_data[g_i] = torch.stack([batches[lut[r, 1]].data[g_i][lut[r, 2]] for r in lut_rows], dim=0)

    # Now form the merged i_x and i_y
    i_x_orig_inds = torch.cat(tuple(b.i_orig[b.i_x] for b in batches))
    i_y_orig_inds = torch.cat(tuple(b.i_orig[b.i_y] for b in batches))

    cat_i_x = torch.Tensor(tuple(torch.nonzero(cat_i_orig == i_o)[0] for i_o in i_x_orig_inds)).long()
    cat_i_y = torch.Tensor(tuple(torch.nonzero(cat_i_orig == i_o)[0] for i_o in i_y_orig_inds)).long()

    # Now form merged TimeSeriesBatch object
    return TimeSeriesBatch(data=cat_data, i_x=cat_i_x, i_y=cat_i_y, i_orig=cat_i_orig)
