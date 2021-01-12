""" Tools for taking different types of averages of data. """

import copy

from typing import Sequence, Tuple

import numpy as np


def aligned_mean(vls: Sequence, align_inds: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """ Aligns time-series data and computes means and standard-errors across time.

    Args:
        vls: vls[i] is the i^th time series of shape [n_smps, n_dims]

        align_inds: align_inds[i] is the index that the data in vls[i] should be aligned to

    Returns:
        mn: The mean values through time of shape [n_tm_pts, n_dims], where n_tm_pts is the number of
        time points covered by all of the aligned data

        std_ers: The standard errors through time of shape [n_tm_pts, n_dims].  If there was only one
        point to calculate the mean at a given point, then the standard error for that point will be nan.

        n_smps: An array of length n_tm_pts. n_smps[t] gives the number of samples that were available
        for computing the mean and standard errors at time point t in mn and std_ers

        ref_aling_ind: The index in the returned mn and std_ers that all time series were aligned to

    Raises:
        ValueError: If each entry in vls is not a 2-d array
    """

    any_vls_not_2d = np.any([v.ndim != 2 for v in vls])
    if any_vls_not_2d:
        raise(ValueError('All data in vls must be two dimensional.'))

    n_vars = vls[0].shape[1]

    align_inds = np.asarray(align_inds).astype('int')

    # ==================================================================================================================
    # Calculate the range of time points we need to represent

    # Max number of time points up to and including alignment index
    max_n_tp_to_ai = int(np.max(align_inds + 1))

    # Max number of times points after alignment index
    max_n_tp_after = int(np.max([vls.shape[0] - a_i - 1 for vls, a_i in zip(vls, align_inds)]))

    n_tm_pts = max_n_tp_to_ai + max_n_tp_after

    # ==================================================================================================================
    # Calculate quantities we need for means and standard errors

    ref_align_ind = max_n_tp_to_ai - 1

    n_smps = np.zeros(n_tm_pts)
    vls_sum = np.zeros([n_tm_pts, n_vars])
    vls_sq_sum = np.zeros([n_tm_pts, n_vars])

    for i, (v_i, a_i) in enumerate(zip(vls, align_inds)):

        first_i = ref_align_ind - a_i
        last_i = first_i + v_i.shape[0]

        n_smps[first_i:last_i] += 1
        vls_sum[first_i:last_i, :] += v_i
        vls_sq_sum[first_i:last_i] += v_i**2

    # ==================================================================================================================
    # Calculate means and standard errors
    vls_sum_t = vls_sum.transpose()
    vls_sq_sum_t = vls_sq_sum.transpose()

    means_t = (vls_sum_t/n_smps)

    # Setup the divisor for calculating standard errors
    std_dev_divisor = copy.deepcopy(n_smps)
    std_dev_divisor = std_dev_divisor - 1
    std_dev_divisor[std_dev_divisor == 0] = 1  # Adjust so we don't  get div. by zero warnings

    std_devs_t = np.sqrt(np.maximum((vls_sq_sum_t - (means_t**2)*n_smps), 0.0)/std_dev_divisor)
    std_ers_t = std_devs_t/np.sqrt(n_smps)

    # Now note where we did not have enough samples to calculate standard error
    std_ers_t[:, n_smps <= 1] = np.nan

    return [means_t.transpose(), std_ers_t.transpose(), n_smps, ref_align_ind]
