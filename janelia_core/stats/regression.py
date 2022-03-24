""" Tools for computing statistics in various ways when working with linear regression models.

The tools provided here are designed for use when assumptions underlying standard approaches to computing statistics
for linear regression models do not apply.  In particular, there are multiple methods for handling grouped errors as
well as methods for computing statistics via boostrap methods.

"""

import copy
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def corr(x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
    """ Calculates the pearson correlation between two series of data.

    If the data is multi-dimensional, the pearson correlation is calculated for each dimension.

    Args:
        x: One set of data of shape n_smps*n_dim

        y: The other set of data; same shape as x

    Returns:
        pearson_corr: The pearson correlation between x and y.  If x & y are multidimensional, then pearson_corr
        is an array and pearson_corr[i] is the correlation for dimension i.
    """

    dims_expanded = False
    if x.ndim == 1:
        dims_expanded = True
        x = np.expand_dims(x, 1)
        y = np.expand_dims(y, 1)

    n_vars = x.shape[1]
    pearson_corr = np.zeros(n_vars)
    for v_i in range(n_vars):
        p_c, _ = scipy.stats.pearsonr(x[:, v_i], y[:, v_i])
        pearson_corr[v_i] = p_c

    if dims_expanded:
        pearson_corr = pearson_corr[0]

    return pearson_corr


def normalized_r_squared(truth: np.ndarray, pred:np.ndarray) -> float:
    """ Computes normalized r-squared for a collection of variables.

    Normalized r-squared is defined as follows:  1 - ( \sum r_i) / ( \sum s_i), where:

        r_i is the squared error between predictions and truth for variable i and s_i is the squared error
        of variable i between the mean value and truth.

    Normalized squared error is equal to standard r-squared when there is only one variable.  When there
    is more than one variable but all variables have the same variance, then normalized r-squared is equal
    to the average of the r-squared values for each individual variable.  However, when some variables have
    higher variance than others, normalized r-squared weights those with higher variance more than those with
    less variance.

    Args:
        truth: True data of shape n_smps*n_vars

        pred: Predicated data of shape n_smps*n_vars
        
    Returns:
        norm_r_sq: The normlaized r-squared

    """

    true_mns = np.mean(truth, 0)
    true_ss = np.sum((truth - true_mns)**2, 0)
    residual_ss = np.sum((truth - pred)**2, 0)

    return 1 - sum(residual_ss)/sum(true_ss)


def r_squared(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """ Computes the r-squared value for a collection of variables.

    Computes the proportion of variance predicted variables capture of ground truth variables.

    Args:
        truth: True data of shape n_smps*n_vars

        pred: Predicated data of shape n_smps*n_vars

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


def grouped_linear_regression_boot_strap(y: np.ndarray, x: np.ndarray, g: np.ndarray, n_bs_smps: int, include_mean: bool = True,
                                         rcond: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Fits a linear regression model, performing a grouped bootstrap to get confidence intervals on coefficients.

    By grouped bootstrap, we mean samples are selected in groups.

    This function returns the coefficients for each bootstrap sample.  Use grouped_linear_regression_boot_strap_stats
    to compute statistics (such as confidence intervals on the coefficients) and visualize_boot_strap_results to
    visualize a summary of the fit results.

    Args:
        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        g: 1-d array indicating groups of samples.  Of length n_smps.  Samples from the same group should
        have the same value in g.

        n_bs_smps: The number of boot strap samples to draw

        include_mean: True if models should have a mean term included.  False if not.

        rcond: The value of rcond to provide to the least squares fitting.  See np.linalg.lstsq.

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
    beta = np.linalg.lstsq(x_aug, y, rcond=rcond)
    beta = beta[0]

    # Perform bootstrap
    bs_beta = np.zeros([n_bs_smps, len(beta)])

    for b_i in range(n_bs_smps):

        # Form sample
        smp_grp_inds = np.random.choice(np.arange(n_grps), n_grps, replace=True)
        bs_inds = np.concatenate([grp_inds[smp_g_i] for smp_g_i in smp_grp_inds])

        bs_x_aug = x_aug[bs_inds, :]
        bs_y = y[bs_inds]

        # Estimate beta for bootstrap sample
        beta_est = np.linalg.lstsq(bs_x_aug, bs_y, rcond=rcond)
        bs_beta[b_i, :] = beta_est[0]

    return [bs_beta, beta]


def grouped_linear_regression_wild_bootstrap(y: np.ndarray, x: np.ndarray, g: np.ndarray,
                                             test_coefs: Sequence[int] = None, n_bs_smps:int = 1000,
                                             rcond: float = None) -> np.ndarray:
    """ Computes a linear model and stats using a wild bootstrap.

    For group g, the model for the i^th observation is of the form:

        y_gi = x_gi^T\beta + o_g + \ep_gi,

    where x_gi are predictor variables of dimension P, o_g is a group offset and ep_gi is zero-mean noise.  In this
    model, it assumes all groups shared the same \beta but each group gets its own o_g and \ep_gi.

    This function will estimate beta as well as p-values that individual coefficients in beta are non-zero.

    The bootstrap procedure is based on the "Wild Cluster bootstrap-t with H0 imposed" bootstrap described in:

        "Boostrap-based Improvements for Inference with Clustered Errors"by Cameron, Gelback & Miller, 2008

    Within each boostrap iteration, the standard-error needed for calculating Wald statistics is estimated with the
    within estimator of:

        "Computing Robust Standard Errors for Within-groups Estimators" by M. Arellano, 1987,

    where a small-sample correction is also applied as recommended in:

        "A Practioner's Guide to Cluster-Robust Inference" by A. Cameron and Douglas Miller, 2015.

    Args:
        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        g: 1-d array indicating groups of samples.  Of length n_smps.  Samples from the same group should
        have the same value in g.

        test_coefs: If None, statistical significant that each individual entry in \beta is different than 0 will be
        calculated.  If not, this is a list of indices of beta to test for

        n_bs_smps: The number of bootstrap samples to use when calculating p-values.

        rcond: The value of rcond to provide to the least squares fitting within the call to
        grouped_linear_regression_within_estimator.  See that function as well np.linalg.lstsq.

    Returns:
        p_vls: The computed p-values for rejecting the null hypothesis that each coefficient is 0.
    """

    n_x_vars = x.shape[1]

    # Determine where the groups are
    grps = np.unique(g)
    n_grps = len(grps)
    grp_inds = [None]*n_grps
    for g_i, grp_i in enumerate(grps):
        grp_inds[g_i] = np.nonzero(g == grp_i)[0]

    # Compute beta
    beta, avm, offsets, _ = grouped_linear_regression_within_estimator(y=y, x=x, g=g, rcond=rcond)

    # Calculate p-values that each individual coefficient in beta is different than 0
    if test_coefs is None:
        test_coefs = np.arange(n_x_vars)

    def coef_p_vl_bs(coef_i, null_vl, n_bs_smps):

        # Compute wald statistic for the observed data
        w = (beta[coef_i] - null_vl)/np.sqrt(avm[coef_i, coef_i])

        # Construct the beta for the null hypothesis
        null_beta = copy.deepcopy(beta)
        null_beta[coef_i] = null_vl

        # Calculate residuals under this null hypothesis
        null_pred = np.matmul(x, null_beta)
        residuals = y - null_pred

        # Get bootstrap samples from distribution of the wald statistic under the null
        w_bs = np.zeros(n_bs_smps)
        for b_i in range(n_bs_smps):

            grp_signs = np.random.choice([-1, 1], n_grps)
            for g_i, g_inds in enumerate(grp_inds):
                residuals[g_inds] = grp_signs[g_i]*residuals[g_inds]
            y_bs = residuals + null_pred
            beta_bs, avm_bs, _, _ = grouped_linear_regression_within_estimator(y=y_bs, x=x, g=g, rcond=rcond)
            w_bs[b_i] = (beta_bs[coef_i] - null_vl)/np.sqrt(avm_bs[coef_i, coef_i])

        # Calculate p-value
        return np.sum(np.abs(w) < np.abs(w_bs))/n_bs_smps

    p_vls = np.zeros(len(test_coefs))
    for c_i, coef_i in enumerate(test_coefs):
        p_vls[c_i] = coef_p_vl_bs(coef_i=coef_i, null_vl=0, n_bs_smps=n_bs_smps)

    return p_vls


def grouped_linear_regression_ols_estimator(y: np.ndarray, x: np.ndarray, g: np.ndarray,
                                            rcond: float = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """ Fits a linear model and stats using optimal least squares, accounting for grouped errors.

     For group g, the model for the i^th observation is of the form:

        y_gi = x_gi^T\beta + \ep_gi,

    where x_gi are predictor variables of dimension P, and ep_gi is noise.  In this model, it assumes all groups shared
    the same \beta but each group gets its own \ep_gi.

    Note: A small sample correction is applied when calculating the asymptotic covariance matrix, as outlined in

        "A Practioner's Guide to Cluster-Robust Inference" by A. Cameron and Douglas Miller, 2015.

    Args:
        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        g: 1-d array indicating groups of samples.  Of length n_smps.  Samples from the same group should
        have the same value in g.

        rcond: The value of rcond to provide to the least squares fitting.  See np.linalg.lstsq.

    Returns:
        beta: The estimate of beta

        acm: The asymptotic covariance matrix for beta.

        n_grps: The number of groups in the analysis

    Raises:
        ValueError: If the number of samples does not exceed the number of x variables.
   """

    n_smps, n_x_vars = x.shape

    if n_x_vars >= n_smps:
        raise(ValueError('Number of samples does not exceed number of x variables.'))

    # Determine where the groups are
    grps = np.unique(g)
    n_grps = len(grps)
    grp_inds = [None]*n_grps
    for g_i, grp_i in enumerate(grps):
        grp_inds[g_i] = np.nonzero(g == grp_i)[0]

    # Calculate beta
    beta_est = np.linalg.lstsq(x, y, rcond=rcond)
    beta = beta_est[0]

    # Calculate asymptotic covariance matrix
    residual = y - np.matmul(x, beta)

    c = (n_grps/(n_grps-1))*((n_smps - 1)/(n_smps - n_x_vars + 1))

    m0 = np.linalg.inv(np.matmul(x.transpose(), x))
    m1 = np.zeros([n_x_vars, n_x_vars])
    for inds in grp_inds:
        temp = np.sqrt(c)*np.matmul(x[inds, :].transpose(), residual[inds])
        temp = temp[:, np.newaxis]
        m1 = m1 + np.matmul(temp, temp.transpose())

    acm = np.matmul(np.matmul(m0, m1), m0)

    return [beta, acm, n_grps]


def grouped_linear_regression_within_estimator(y: np.ndarray, x: np.ndarray, g: np.ndarray,
                                               rcond: float = None) -> Tuple[np.ndarray, np.ndarray, dict, int]:
    """ Computes linear model and stats using the within estimator for a fixed-effects linear model.

    For group g, the model for the i^th observation is of the form:

        y_gi = x_gi^T\beta + o_g + \ep_gi,

    where x_gi are predictor variables of dimension P, o_g is a group offset and ep_gi is zero-mean noise.  In this model,
    it assumes all groups shared the same \beta but each group gets its own o_g and \ep_gi.

    This function will estimate beta as well as the asymptotic covariance matrix for beta using the method of:

        "Computing Robust Standard Errors for Within-groups Estimators" by M. Arellano, 1987.

    It will then apply a finite sample correction as recommended in:

        "A Practioner's Guide to Cluster-Robust Inference" by A. Cameron and Douglas Miller, 2015.

    Args:
        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        g: 1-d array indicating groups of samples.  Of length n_smps.  Samples from the same group should
        have the same value in g.

        rcond: The value of rcond to provide to the least squares fitting.  See np.linalg.lstsq.

    Returns:
        beta: The estimate of beta

        acm: The asymptotic covariance matrix for beta.

        offsets: A dictionary with the o_g values for each group.  Keys will be values of g used to indicate groups and
        values will be the offset for each group.

        n_grps: The number of groups in the analysis
   """

    n_smps, n_x_vars = x.shape

    # Determine where the groups are
    grps = np.unique(g)
    n_grps = len(grps)
    grp_inds = [None]*n_grps
    for g_i, grp_i in enumerate(grps):
        grp_inds[g_i] = np.nonzero(g == grp_i)[0]

    # Mean center x and y
    x_ctr = copy.deepcopy(x)
    y_ctr = copy.deepcopy(y)
    for inds in grp_inds:
        x_ctr[inds, :] = x_ctr[inds, :] - np.mean(x_ctr[inds, :], axis=0)
        y_ctr[inds] = y_ctr[inds] - np.mean(y_ctr[inds])

    # Calculate beta
    beta_est = np.linalg.lstsq(x_ctr, y_ctr, rcond=rcond)
    beta = beta_est[0]

    # Calculate the asymptotic variance matrix for beta
    residual = y_ctr - np.matmul(x_ctr, beta)

    c = (n_grps/(n_grps - 1))*((n_smps-1)/(n_smps - n_x_vars + 1))  # Correction factor for finite samples

    m0 = np.linalg.inv(np.matmul(x_ctr.transpose(), x_ctr))
    m1 = np.zeros([n_x_vars, n_x_vars])
    for inds in grp_inds:
        temp = np.sqrt(c)*np.matmul(x_ctr[inds, :].transpose(), residual[inds])
        temp = temp[:, np.newaxis]
        m1 = m1 + np.matmul(temp, temp.transpose())

    plt.figure()
    plt.imshow(m0)
    plt.colorbar()
    plt.title('m0')

    plt.figure()
    plt.imshow(m1)
    plt.colorbar()
    plt.title('m1')

    acm = np.matmul(np.matmul(m0, m1), m0)

    # Calculate offsets for each group
    uncentered_res = y - np.matmul(x, beta)
    o_g = np.zeros(n_grps)
    for g_i, g_inds in enumerate(grp_inds):
        o_g[g_i] = np.mean(uncentered_res[g_inds])

    offsets = {g:v for g, v in zip(grps, o_g)}

    return [beta, acm, offsets, n_grps]


def grouped_linear_regression_acm_stats(beta, acm, n_grps, alpha) -> dict:
    """ Calculates statistics given an estimate of an asymptotic covariance matrix.

    Confidence intervals and p-values for individual coefficients are calculated assuming a t-distribution on the
    estimates for the individual entries of beta.

    Args:
        beta: The estimate of beta

        acm: The asymptotic variance matrix for beta

        n_grps: The number of groups in the original regression

        alpha: The alpha value to use when constructing 1-alpha confidence intervals

    Returns:
        stats: A dictionary with the following keys:

            alpha: The alpha value for which confidence intervals were calculated.

            c_ints: Confidence intervals.  c_ints[:,i] is the 1-alpha percentile confidence interval for beta[i]

            non_zero_p: Indicates the p-value for null hypothesis that beta[i] is 0.

            non_zero: non_zero[i] is true if the confidence interval for beta[i] does not contain 0.
    """

    m = -1*scipy.stats.t(df=(n_grps - 1)).ppf(alpha / 2)
    std_ers = np.sqrt(np.diag(acm))

    c_ints = np.stack([beta - std_ers*m, beta+std_ers*m])

    non_zero_p = 2*scipy.stats.t(df=(n_grps-1)).cdf(-1*np.abs(beta/std_ers))

    non_zero = np.logical_not(np.logical_and(c_ints[0,:] < 0, c_ints[1,:] > 0))

    return {'alpha': alpha, 'c_ints': c_ints, 'non_zero_p': non_zero_p, 'non_zero': non_zero}


def grouped_linear_regression_acm_linear_restriction_stats(beta: np.ndarray, acm: np.ndarray, r: np.ndarray,
                                                           q: np.ndarray, n_grps: int) -> float:
    """ Given an asymptotic covariance matrix from linear regression, tests significance of linear restrictions.

    Given a vector of K-coefficients, beta, we desire to produce a p-value for a null hypothesis taking the form of
    J linear restrictions, which can be written as:

    r*beta = q, where r is a matrix of size J by K and q is a vector of length J.

    This function was based heavily on chapter 5 of Greene, Econometric Analysis.  In addition, to this
    reference a good reference on Wald tests (the Wikipedia article is not a bad place to start) may also be
    useful.

    The basic approach we take here is we assume:

    1) The distribution on beta is well approximated as a normal distribution with the asymptotic covariance matrix
    provided to this function.

    2) Under this assumption, we can compute the F-statistic (see Greene), which is defined as:

        F = [r*beta - q]' (r' * acm * r)^{-1} [r*beta - q]/J,

        which has a distribution equal to 1/J times a Chi-squared distribution with K degrees of freedom.

    3) We can in principle calculate critical values comparing F*J to critical values from a Chi-square distribution.
    However, again following Greene, we choose to use critical values from the F-distribution, which is more
    conservative.  Specifically, we compare F (no need now to multiply by J) to the critical values from the
    F(J, n_grps - K) distribution.

    Note: This function can be used for computing statistics for linear restrictions of non-grouped results as well.
    In this case, n_grps should just be seq equal to the number of samples in the data.

    Args:
        beta: The beta vector which was estimated

        acm: The estimated asymptotic covariance matrix

        r: The r matrix for linear restrictions.  Each row is a restriction.

        q: The q vector for linear restrictions.

        n_grps: The number of groups in the original data

    Returns:
        p: The calculated p value.

    """

    # Expand beta
    beta = np.expand_dims(beta, 1)

    # Make sure r is always 2-d, even if user only gave one linear restriction.
    if r.ndim == 1:
        r = np.expand_dims(r, 0)

    # Make sure q is always a vector
    if q.ndim == 1:
        q = np.expand_dims(q, 1)

    K = len(beta)
    J = r.shape[0]

    diff = np.matmul(r, beta) - q
    test_acm = np.matmul(np.matmul(r, acm), r.T)

    f = (np.matmul(np.matmul(diff.T, np.linalg.inv(test_acm)), diff)/J).squeeze()

    return 1 - scipy.stats.f.cdf(f, J, n_grps - K)


def naive_regression(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Performs "naive" regression to predict x from y.

    By naive regression, we assume that all x variables are mutually independent. The parameters this
    function provides is the correct regression model for this case.  If not all variables are
    independent, the parameters returned by this function are sub-optimal.

    This function will return the parameters, b (a matrix) and o (a vector of offsets) to predict:

    y_i = x_i^T*b + o

    where we asume y_i \in R^m and x_i \in R^p.

    Args:
        y: Data to predict of shape [n_smps, m]

        x: Data to predict from of shape [n_smps, p]

    Returns:
        b: the b matrix above

        o: the offset vector above

    """

    if (x.ndim != 2) or (y.ndim != 2):
        raise(ValueError('x and y must be 2-d arrays'))

    n_smps, n_x_vars = x.shape
    n_y_vars = y.shape[1]

    # Calculate means
    mu = np.mean(x, axis=0)
    y_mn = np.mean(y, axis=0)

    var = np.var(x, axis=0)

    x_ctr = x - mu

    b = np.zeros([n_x_vars, n_y_vars])
    for v_i in range(n_y_vars):
        b[:, v_i] = np.sum((y[:, v_i:v_i+1]*x_ctr)/(var*n_smps), axis=0)

    o = y_mn - np.matmul(b.transpose(), mu)

    return (b, o)


def visualize_boot_strap_results(bs_values: np.ndarray, var_strs: Sequence, theta: np.ndarray = None,
                                 var_clrs: np.ndarray = None, violin_plots: bool = True,
                                 plot_c_ints: bool = True, show_nz_sig: bool = True,
                                 plot_zero_line:bool = True,
                                 alpha: float = .05, theta_size: int = 5,
                                 er_bar_pts: int = 2, sig_size:int = 5, sig_y_vl: float = None,
                                 ax:plt.axes = None) -> plt.axes:

    """ For visualizing the results of grouped_linear_regression_boot_strap.

    Args:
        bs_values: The coefficients for each bootstrap sample. bs_values[i,:] are the coefficients for bootstrap sample
        i.

        var_strs: The names of each of the coefficients in bs_values.

        theta: Point estimates for each coefficient.  If None, point values will not be plotted.

        var_clrs: Of shape n_coefs*3.  var_clrs[i,:] is a color for plotting the values associated with coefficient i.

        violin_plots: If true, violin plots of coefficient values will be plotted.

        plot_c_ints: True if confidence intervals for each coefficient should be plotted.

        plot_zero_line: True if a dotted line denoting 0 should be added to the plot.

        show_nz_sig: True if coefficients significantly different than 0 should be denoted with stars. This is
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

    # Set colors of x-axis labels
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


def grouped_linear_regression_boot_strap_stats(bs_values: np.ndarray, alpha:float =.05) -> dict:
    """ For getting statistics from the results of grouped_linear_regression_boot_strap.

    This function will compute:

        1) Confidence intervals for each coefficient.  Currently percentile confidence intervals are computed. See
        "All of Statistics" by Wasserman for more information on percentile confidence intervals for the bootstrap.

        2) P-values that coefficients are significantly different than 0, based on inverting percentile confidence
        intervals

    Args:
        bs_values: The results of grouped_linear_regresson_boot_strap.  bs_vls[i,:] are the coefficient for
        the i^th bootstrap sample.

        alpha: The alpha values to use when computing confidence intervals

    Returns:
        stats: A dictionary with the following keys:

            alpha: The alpha value for which confidence intervals were calculated.

            c_ints: Confidence intervals.  c_ints[:,i] is the 1-alpha percentile confidence interval for coefficient i.

            non_zero_p: Indicates the p-value for null hypothesis that the coefficient is 0.

            non_zero: Indicates coefficients with 1-alpha confidence intervals which do not contain 0.  non_zero[i] is
            true if the confidence interval for coefficient i does not contain 0.

    """

    # Calculate percentile confidence intervals
    n_smps, n_coefs = bs_values.shape
    c_ints = np.zeros([2, n_coefs])
    c_ints[0, :] = np.percentile(bs_values, q=100*alpha/2, axis=0)
    c_ints[1, :] = np.percentile(bs_values, q=100*(1 - alpha/2), axis=0)

    # See which confidence intervals do not contain zero
    non_zero = np.logical_not(np.logical_and(c_ints[0,:] <= 0, c_ints[1,:] >= 0))

    # =============================================================================================
    # Calculate p-values
    # =============================================================================================

    # Find smallest values greater than 0 for each coefficient
    sm_search_vls = copy.copy(bs_values)
    sm_search_vls[sm_search_vls < 0] = np.inf
    sm_values = np.min(sm_search_vls, axis=0)

    # Find largest value less than 0 for each coefficient
    lg_search_values = copy.copy(bs_values)
    lg_search_values[lg_search_values > 0] = -np.inf
    lg_values = np.max(lg_search_values, axis=0)

    # Get percentage of entries larger and smaller than values above
    sm_p = np.sum(bs_values <= sm_values, axis=0)/n_smps
    lg_p = np.sum(bs_values >= lg_values, axis=0)/n_smps

    non_zero_p = 2*np.min(np.column_stack([sm_p, lg_p]), axis=1)

    # Return results
    return {'alpha': alpha, 'c_ints': c_ints, 'non_zero_p': non_zero_p, 'non_zero': non_zero}


def linear_regression_ols_estimator(y: np.ndarray, x: np.ndarray, rcond: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Fits a linear model and stats using optimal least squares.

     For group g, the model for the i^th observation is of the form:

        y_i = x_i^T\beta + \ep_i,

    where x_i are predictor variables of dimension P, and ep_i is noise with a fixed variance.

    This function is based on chapter 4 of Greene, Econometric Analysis.

    Args:
        y: 1-d array of the predicted variable.  Of length n_smps.

        x: Variables to predict from.  Of shape n_smps*d_x.

        rcond: The value of rcond to provide to the least squares fitting.  See np.linalg.lstsq.

    Returns:
        beta: The estimate of beta

        acm: The asymptotic covariance matrix for beta.

    Raises:
        ValueError: If the number of samples does not exceed the number of x variables.
   """

    n_smps, n_x_vars = x.shape

    if n_x_vars >= n_smps:
        raise(ValueError('Number of samples does not exceed number of x variables.'))

    # Calculate beta
    beta_est = np.linalg.lstsq(x, y, rcond=rcond)
    beta = beta_est[0]

    # Calculate the asymptotic variance matrix
    Q = np.matmul(x.transpose(), x)  # This is actually 1/n * Q
    er = y - np.matmul(x, beta)
    s_sq = np.sum(er**2)/(n_smps - n_x_vars)

    acm = s_sq*np.linalg.inv(Q)

    return beta, acm


def visualize_coefficient_stats(var_strs: Sequence, theta: np.ndarray = None, c_ints: np.ndarray = None,
                                var_clrs: np.ndarray = None, sig: np.ndarray = None,
                                er_bar_pts: int = 2, theta_size = 5,
                                sig_size:int = 5, sig_y_vl: float = None, x_axis_rot: float = -75,
                                ax: plt.axes = None, plot_zero_line: bool = True) -> plt.axes:

    """ For visualizing estimated coefficients with confidence intervals.

    Args:
        var_strs: Strings giving the name of each coefficient

        theta: Point estimate for each coefficient, order corresponds to var_strs.  If None, point estimates will
        not be plotted.

        c_ints: c_ints[i,:] is the lower (left column) and upper (right column) limits on the confidence interval
        for the coefficient in var_strs[i].  If None, confidencce intervals will not be plotted.

        var_clrs: var_clrs[i,:] is the color to use for plotting the coefficient for var_strs[i].  If None, black
        will be used for all plotting.

        sig: sig[i] is True if the coefficient for var_strs[i] is significant. Significant values will have a
        star plotted to indicate they are significant.

        theta_size: The size of the marker to use when plotting point estimages of coefficients.

        er_bar_pts: The width of error bars to plot for confidence intervals.

        sig_size: The size of the marker to use when denoting which coefficients are significanly different 0.

        sig_y_vl: If not None, this is the y-value used for showing significant stars.

        plot_zero_line: True if a line indicating 0 should be plotted

        x_axis_rot: Rotation in degrees of x-axis labels.

        ax: Axes to plot into. If None, a new figure with axes will be created.

    Returns:
        as: The axes the plot was generated in.
    """

    n_vars = len(var_strs)

    if ax is None:
        plt.figure()
        ax = plt.axes()

    if var_clrs is None:
        var_clrs = np.zeros([n_vars, 3])

    # Label x-axis
    plt.xticks(np.arange(1, n_vars+1), var_strs, rotation=x_axis_rot)

    # Set colors of x-axis labels
    for var_i, x_lbl in enumerate(ax.get_xticklabels()):
        x_lbl.set_color(var_clrs[var_i, :])

    # Plot zero line if we are suppose to
    if plot_zero_line:
        plt.plot([0, n_vars+1], [0, 0], 'k--')

    # Plot confidence intervals if we are suppose to
    if c_ints is not None:
        err_y = np.mean(c_ints, axis=0)
        lower_err_bars = err_y - c_ints[0, :]
        upper_err_bars = c_ints[1, :] - err_y
        err_bars = np.stack([lower_err_bars, upper_err_bars])

        for v_i in range(n_vars):
            ax.errorbar(x=v_i+1, y=err_y[v_i], yerr=err_bars[:, v_i].reshape(2,1), fmt='none',
                        ecolor=var_clrs[v_i,:], elinewidth=er_bar_pts,
                        capsize=2*er_bar_pts, capthick=er_bar_pts)

    # Show estimated point values if we are suppose to
    if theta is not None:
        for v_i in range(n_vars):
            plt.plot(v_i + 1, theta[v_i], 'ko', markersize=theta_size, color=var_clrs[v_i,:])

    # Add asteriks to indicate significant if we are suppose to
    if sig is not None:
        y_lim = ax.get_ylim()
        if sig_y_vl is None:
            sig_y_vl = .02*(y_lim[1] - y_lim[0]) + y_lim[0]
        for v_i in range(n_vars):
            if sig[v_i]:
                plt.plot(v_i+1, sig_y_vl, 'k*', markersize=sig_size, color=var_clrs[v_i,:])

    return ax


