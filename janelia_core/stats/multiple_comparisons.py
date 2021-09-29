"""  Tools for controlling for multiple comparisons.  """

from typing import Tuple

import numpy as np


def apply_by(p_vls: np.ndarray, alpha: float) -> Tuple[np.ndarray]:
    """ Applies the Benjamini-Yekutieli procedure to control the false discovery rate.

    See the paper "THE CONTROL OF THE FALSE DISCOVERY RATE IN MULTIPLE TESTING UNDER DEPENDENCY"
    by Benjamini and Yekutieli for more details on the Benjamini-Yekutieli procedure.

    See the paper "Resampling-based false discovery rate controlling multiple test procedures for correlated test
    statistics" by Yekutieli and Benjamini for background on FDR adjusted p values.

    Args:

        p_vls: The p values for the null hypothesis to control.  Can be an array of any size.

        alpha: The level to control the false discovery rate at.

    Return:

        reject_tests: A binary array.  True values correspond to entries in p_vls that should be
        rejected.

        adjusted_p_vls: The array of FDR adjusted p-values.

    """

    orig_shape = p_vls.shape
    flat_p_vls = np.ravel(p_vls)
    sort_order = np.argsort(flat_p_vls)
    sorted_p_vls = flat_p_vls[sort_order]
    n_tests = len(sorted_p_vls)

    # Calculate corrected p values
    c = np.sum(1/np.arange(1, n_tests+1))
    correction_f = (np.arange(1, n_tests+1) / n_tests) / c
    sorted_adjusted_p_vls = sorted_p_vls/correction_f
    for i in range(len(sorted_p_vls)):
        sorted_adjusted_p_vls[i] = np.min(sorted_adjusted_p_vls[i:])
    sorted_adjusted_p_vls[sorted_adjusted_p_vls > 1] = 1

    # Determine which tests we can reject
    reject_inds = sorted_adjusted_p_vls <= alpha

    # Return output in original shape and ordering
    unsorted_reject_tests = np.zeros(sorted_p_vls.shape)
    unsorted_adjusted_p_vls = np.zeros(sorted_p_vls.shape)

    unsorted_reject_tests[sort_order] = reject_inds
    unsorted_adjusted_p_vls[sort_order] = sorted_adjusted_p_vls

    reshaped_reject_tests = np.reshape(unsorted_reject_tests, orig_shape)
    reshaped_adjusted_p_vls = np.reshape(unsorted_adjusted_p_vls, orig_shape)

    return reshaped_reject_tests, reshaped_adjusted_p_vls


def apply_bonferroni(p_vls: np.ndarray, alpha: float) -> Tuple[np.ndarray]:
    """ Applies a Bonferroni correction to p values.

    Args:

        p_vls: An array of p values.  Can be any shape.

        alpha: The level to test for significance at.  Between 0 and 1, inclusive.

    Returns:

        rejected_tests: A boolean array, the same shape as p_vls, indicating which null hypothesises should be rejected

        adjusted_p_vls: Bonferoni adjusted p-values.

    """

    adjusted_p_vls = p_vls*p_vls.size
    rejected_tests = adjusted_p_vls <= alpha

    return rejected_tests, adjusted_p_vls

