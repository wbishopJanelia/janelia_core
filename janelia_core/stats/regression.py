""" Tools for regression.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


def r_squared(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """ Computes the r-squared value for a collection of variables.

    Computes proportion of variance predicted variables capture of ground truth variables.
    Args:
        truth: True data of shape n_smps*n_vars

        pred: Prdicated data of shape n_smps*n_vars

    Returns:

        r_sq: r_sq[i] contains the r-squared value for pred[:, i]
    """


    if pred.ndim == 1:
        pred = np.expand_dims(pred, 1)
        truth = np.expand_dims(truth, 1)

    true_mns = np.mean(truth, 0)
    true_ss = np.sum((truth - true_mns)**2, 0)
    residual_ss = np.sum((truth - pred)**2, 0)

    r_sq = np.squeeze(1 - residual_ss/true_ss)
    return r_sq




