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

    This function returns the coefficients for each boot-strap sample.  Use grouped_linear_regression_boot_strap_stats
    to compute statistics (such as confidence intervals on the coefficients) and visualize_boot_strap_results to
    visualize a summary of the fit results.

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
        smp_grp_inds = np.random.choice(np.arange(n_grps), n_grps, replace=True)
        bs_inds = np.concatenate([grp_inds[smp_g_i] for smp_g_i in smp_grp_inds])

        #bs_inds = np.concatenate(np.random.choice(grp_inds, n_grps, replace=True))
        bs_x_aug = x_aug[bs_inds, :]
        bs_y = y[bs_inds]

        # Estimate beta for bootstrap sample
        beta_est = np.linalg.lstsq(bs_x_aug, bs_y, rcond=None)
        bs_beta[b_i, :] = beta_est[0]

    return [bs_beta, beta]


def visualize_boot_strap_results(bs_values: np.ndarray, var_strs: Sequence, theta: np.ndarray = None,
                                 var_clrs: np.ndarray = None, violin_plots: bool = True,
                                 plot_c_ints: bool = True, show_nz_sig: bool = True,
                                 plot_zero_line:bool = True,
                                 alpha: float = .05, theta_size: int = 5,
                                 er_bar_pts: int = 2, sig_size:int = 5, sig_y_vl: float = None,
                                 ax:plt.axes = None) -> plt.axes:

    """ For visualizing the results of grouped_linear_regression_boot_strap.

    Args:
        bs_values: The coefficients for each bootstrap sample. bs_values[i,:] are the coefficients for bootstramp sample
        i.

        var_strs: The names of each of the coefficients in bs_values.

        theta: Point estimates for each coefficient.  If None, point values will not be plotted.

        var_clrs: Of shape n_coefs*3.  var_clrs[i,:] is a color for plotting the values associated with coefficient i.

        violin_plots: If true, violin plots of coefficient values will be plotted.

        plot_c_ints: True if confidence intervals for each coefficient should be plotted.

        plot_zero_line: True if a dotted line denoting 0 should be added to the plot.

        show_nz_sig: True if coefficients significantly different than 0 should be denoted with starts. This is
        computed simply by testing if the confidence interval for a coefficient contains 0.

        alpha: The alpha value for constructing confidence intervals and determining if coefficients are significantly
        different than 0.

        theta_size: The size of the marker to use when plotting point estimages of coefficients.

        er_bar_pts: The width of error bars to plot for confidence intervals.

        sig_size: The size of the marker to use when denoting which coefficients are significanly different 0.

        sig_y_vl: If not None, this is the y-value used for showing significant stars.

        ax: The axis to plot into.  If none, one will be created.

    Returns:
        ax: The axes the plot was created in

    """

    n_vars = len(var_strs)

    if ax is None:
        plt.figure()
        ax = plt.axes()

    if var_clrs is None:
        var_clrs = np.zeros([n_vars, 3])

    # Create violin plots if we are suppose to
    if violin_plots:
        vp_parts = ax.violinplot(bs_values, showmeans=False, showmedians=False, showextrema=False)
        for var_i, body in enumerate(vp_parts['bodies']):
            body.set_facecolor(var_clrs[var_i, :])

    # Label x-axis
    plt.xticks(np.arange(1, n_vars+1), var_strs, rotation=-75)

    # Set colors of x-axix labels
    for var_i, x_lbl in enumerate(ax.get_xticklabels()):
        x_lbl.set_color(var_clrs[var_i, :])

    # Plot zero line if we are suppose to
    if plot_zero_line:
        plt.plot([0, n_vars+1], [0, 0], 'k--')

    # Plot statistics if we are suppose to
    if plot_c_ints or show_nz_sig:
        stats = grouped_linear_regression_boot_strap_stats(bs_values=bs_values, alpha=alpha)

        if plot_c_ints:
            err_y = np.mean(stats['c_ints'], axis=0)
            lower_err_bars = err_y - stats['c_ints'][0, :]
            upper_err_bars = stats['c_ints'][1, :] - err_y
            err_bars = np.stack([lower_err_bars, upper_err_bars])

            for v_i in range(n_vars):
                ax.errorbar(x=v_i+1, y=err_y[v_i], yerr=err_bars[:, v_i].reshape(2,1), fmt='none',
                            ecolor=var_clrs[v_i,:], elinewidth=er_bar_pts,
                            capsize=2*er_bar_pts, capthick=er_bar_pts)

        if show_nz_sig:
            y_lim = ax.get_ylim()
            if sig_y_vl is None:
                sig_y_vl = .02*(y_lim[1] - y_lim[0]) + y_lim[0]
            for v_i in range(n_vars):
                if stats['non_zero'][v_i]:
                    plt.plot(v_i+1, sig_y_vl, 'k*', markersize=sig_size, color=var_clrs[v_i,:])

    # Show estimated point values
    if theta is not None:
        for v_i in range(n_vars):
            plt.plot(v_i + 1, theta[v_i], 'ko', markersize=theta_size, color=var_clrs[v_i,:])

    return ax


def grouped_linear_regression_boot_strap_stats(bs_values: np.ndarray, alpha:float =.05):
    """ For getting statistics from the results of grouped_linear_regression_boot_strap.

    This function will compute:

        1) Confidence intervals for each coefficient.  Currently percentile confidence intervals are computed. See
        "All of Statistics" by Wasserman for more information on percentile confidence intervals for the bootstrap.


    Args:
        bs_values: The results of grouped_linear_regresson_boot_strap.  bs_vls[i,:] are the coefficient for
        the i^th bootstrap sample.

        alpha: The alpha values to use when computing confidence intervals

    Returns:

        stats: A dictionary with the following keys:

            alpha: The alpha value for which statistics were calculated.

            c_ints: Confidence intervals.  c_ints[:,i] is the 1-alpha percentile confidence interval for coefficient i.

            non_zero: Indicates coefficients with 1-alpha confidence intervals which do not contain 0.  non_zero[i] is
            true if the confidence interval for coefficient i does not contain 0.
    """

    # Calculate percentile confidence intervals
    n_coefs = bs_values.shape[1]
    c_ints = np.zeros([2, n_coefs])
    c_ints[0, :] = np.percentile(bs_values, q=100*alpha/2, axis=0)
    c_ints[1, :] = np.percentile(bs_values, q=100*(1 - alpha/2), axis=0)

    # See which confidence intervals do not contain zero
    non_zero = np.logical_not(np.logical_and(c_ints[0,:] <= 0, c_ints[1,:] >= 0))

    return {'alpha': alpha, 'c_ints': c_ints, 'non_zero': non_zero}


