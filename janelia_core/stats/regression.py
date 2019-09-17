""" Tools for regression.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

import matplotlib.pyplot as plt
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


def grouped_linear_regression_boot_strap(y: np.ndarray, x: np.ndarray, g: np.ndarray, n_bs_smps: int, include_mean: bool = True):
    """ Fits a linear regression model, performing a grouped bootstrap to get confidence intervals on coefficients.

    By grouped boot-strap, we mean samples are selected in groups.

    Args:

        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        g: 1-d array indicating groups of samples.  Of length n_smps.  Samples from the same group should
        have the same value in g.

        n_bs_smps: The number of boot strap samples to draw

        include_mean: True if models should have a mean term included.  False if not.

    Returns:

        bs_beta: Of shape n_boot_strap_smps*n_coefs.  bs_beta[i,:] contains the coefficients of the linear regression
        model for the i^th bootstrap sample.  If include_mean is True, the last column of bs_beta contains the mean.

        beta: The beta fit to the data.  If include_mean is True, the last entry of bs_beta contains the mean.

    """

    n_smps = len(g)

    # Determine where the groups are
    grps = np.unique(g)
    n_grps = len(grps)
    grp_inds = [None]*n_grps
    for g_i, grp_i in enumerate(grps):
        grp_inds[g_i] = np.nonzero(g == grp_i)[0]

    # Add term in x for mean
    if include_mean:
        x_aug = np.concatenate([x, np.ones([n_smps, 1])], axis=1)
    else:
        x_aug = x

    # Estimate beta
    beta = np.linalg.lstsq(x_aug, y, rcond=None)
    beta = beta[0]

    # Perform bootstrap
    bs_beta = np.zeros([n_bs_smps, len(beta)])

    for b_i in range(n_bs_smps):

        # Form sample
        bs_inds = np.concatenate(np.random.choice(grp_inds, n_grps, replace=True))
        bs_x_aug = x_aug[bs_inds, :]
        bs_y = y[bs_inds]

        # Estimate beta for bootstrap sample
        beta_est = np.linalg.lstsq(bs_x_aug, bs_y, rcond=None)
        bs_beta[b_i, :] = beta_est[0]

    return [bs_beta, beta]


def visualize_boot_strap_results(bs_values: np.ndarray, var_strs: Sequence,
                                 var_clrs: np.ndarray = None, violin_plots: bool = True,
                                 ax:plt.axes = None, plot_zero_line:bool = True):

    n_vars = len(var_strs)

    if ax is None:
        plt.figure()
        ax = plt.axes()

    # Create violin plots if we are suppose to
    if violin_plots:
        vp_parts = ax.violinplot(bs_values, showmeans=False, showmedians=False, showextrema=False)
        if var_clrs is not None:
            for var_i, body in enumerate(vp_parts['bodies']):
                body.set_facecolor(var_clrs[var_i, :])

    # Label x-axis
    plt.xticks(np.arange(1, n_vars+1), var_strs, rotation=-75)

    # Set colors of x-axix labels
    if var_clrs is not None:
        for var_i, x_lbl in enumerate(ax.get_xticklabels()):
            x_lbl.set_color(var_clrs[var_i, :])

    # Plot zero line if we are suppose to
    if plot_zero_line:
        plt.plot([1, n_vars+1], [0, 0], 'k--')

