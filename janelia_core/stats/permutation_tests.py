""" Contains tools for performing permutation tests. """

import copy
from typing import Callable, Sequence, Tuple

import numpy as np


def basic_perm_test(x_0: np.ndarray, x_1: np.ndarray, n_perms: int = 1000, f: Callable = None) -> Tuple[float, float]:
    """ Performs a basic permutation test.

    This function calculates the p value for a statistic of the form:

        t = abs(f(x_0) - f(x_1)),

    where f() is be default median().  In this case, the p-value is for the difference of medians of two
    samples.  However, the user can specify custom f functions (e.g., mean) to calculate the p-value
    for other statistics.

    Args:
        x_0: Data from condition 0.  Should be either 1 or 2-d array.  Each row is a sample.

        x_1: Data from condition 1.  Should be either 1 or 2-d array.  Each row is a sample.
        Note: x_1 can have a different number of samples than x_0.

        n_perms: The number of permutations to perform.

        f: The f function to use (see above).  If None, median will be used.  If samples have a dimensionality greater
        than 1, a custom f function must be used.

    Returns:
        p: The p value for the calculated statistic

        t: The calculated statistic

    Raises:
        TypeError: If x_0 or x_1 are not of type np.ndarray
        ValueError: If x_0 and x_1 do not have at least one sample each
        ValueError: If samples have a dimensionality greater than 1 but the default f is used.
    """

    if (not isinstance(x_0, np.ndarray)) or (not isinstance(x_1, np.ndarray)):
        raise(TypeError('x_0 and x_1 must be of type np.ndarray.'))

    # Form x_0 and x_1 into 2-d arrays if they are 1-d
    if x_0.ndim == 1:
        x_0 = np.expand_dims(x_0, 1)
    if x_1.ndim == 1:
        x_1 = np.expand_dims(x_1, 1)

    n_x_0 = x_0.shape[0]
    n_x_1 = x_1.shape[0]

    if (n_x_0 == 0) or (n_x_1 == 0):
        raise(ValueError('There must be at least one sample in x0 and x1.'))

    if f is None:
        if x_0.shape[1] != 1:
            raise(ValueError('Using the default function (median) is only valid for 1-d input.'))
        f = np.median

    # Calculate the test statistic
    t = np.abs(f(x_0) - f(x_1))

    # Perform the permutation tests
    n_total = n_x_0 + n_x_1
    c_data = np.concatenate([x_0, x_1])
    perm_t = np.zeros(n_perms)
    for p_i in range(n_perms):
        perm_inds = np.random.choice(a=n_total, size=n_total, replace=False)
        x_0_perm_inds = perm_inds[0:n_x_0]
        x_1_perm_inds = perm_inds[n_x_0:]
        perm_t[p_i] = np.abs(f(c_data[x_0_perm_inds,:]) - f(c_data[x_1_perm_inds, :]))

    p = np.sum(perm_t >= t)/n_perms

    return [p, t]


def paired_grouped_perm_test(x0: np.ndarray, x1: np.ndarray, grp_ids: np.ndarray, pair_f: Callable = None,
                             reduce_f: Callable = None, n_perms: int = 1000) -> Tuple[float, float]:
    """ Performs a paired, grouped permutation test.

    This calculates paired statistics.  In otherwords, we assume each sample from condition 0 is paired with a sample
    from condition 1.   The user can specify the function (e.g., abs(x1 - x0)) which is used to compare the samples. We
    then test against the null hypothesis that x1 and x0 come from the same distribution.

    This is also for performing permutation tests when samples are grouped.  A good example of this is when we obtain
    multiple samples from one subject. In this case all the samples from one subject will form one group. Because we
    expect that samples in the same group may not be truly independent, we need to take this into account when
    performing permutations and permute all the samples from a group together.  This is what this test does.  It is
    essentially a standard permutation test, except samples in groups are shuffled together.

    Args:
        x0: Array of samples from condition 0.  Each row is a sample.  Samples from all groups are concatenated together.

        x1: Array of samples from condition 1.  Rows of x1 are paired with rows of x0.

        grp_ids: a 1-d array indicating groups of samples.  Samples from the same group should
        have the same value in g.

        pair_f: The function used to compare x0 and x1.  If None, the function x1 - x0 will be used.

        reduce_f: The function used to calculate a single statistic from the results of pair_f.  If None, the function
        mean() will be used.

    Returns:

        t: The value of the test statistic

        p: The p-value
    """

    if pair_f is None:
        pair_f = lambda x, y: y - x

    if reduce_f is None:
        reduce_f = np.mean

    # Determine where the groups are
    grps = np.unique(grp_ids)
    n_grps = len(grps)
    grp_inds = [None]*n_grps
    for g_i, grp_i in enumerate(grps):
        grp_inds[g_i] = np.nonzero(grp_ids == grp_i)[0]

    # Calculate the test statistic
    t = reduce_f(pair_f(x0, x1))

    # Calculate the test statistic for permutations
    perm_t = np.zeros(n_perms)
    for p_i in range(n_perms):
        perm_inds = np.random.binomial(n=1, p=.5, size=n_grps)
        perm_x0 = copy.deepcopy(x0)
        perm_x1 = copy.deepcopy(x1)
        for g_i, grp_ind in enumerate(grp_inds):
            if perm_inds[g_i] == 1:
                perm_x0[grp_ind] = x1[grp_ind]
                perm_x1[grp_ind] = x0[grp_ind]

        perm_t[p_i] = reduce_f(pair_f(perm_x0, perm_x1))

    # Calculate p-value
    p = np.sum(np.abs(perm_t) >= np.abs(t))/n_perms

    return [t, p]


def all_pairs_perm_tests(x: Sequence[np.ndarray], test_opts: dict = None, update_int=10):
    """ Performs a set of permutation tests between all pairs of conditions.

    Args:
        x: x[i] is a numpy array of samples for condition i.  x[i] should either be a 1-d or 2-d numpy array, where
        each row is a sample.  Different entries of x can have different number of samples.

        test_opts: A dictionary of keyword arguments to pass into the call to basic_perm_test for each test.

        update_int: Progress is printed to screen every update_int number of tests.

    Returns:
        p_vls: p_vls[i,j] is the p value when testing the difference between x[i] and x[j].  To save computation, we
        set p_vls[j,i] = p_vls[i,j].  The diagonal of p_vls will be nan.
    """

    if test_opts is None:
        test_opts = dict()

    n_x = len(x)

    n_total_tests = int(.5*(n_x + 1)*n_x - n_x)

    p_vls = np.zeros([n_x, n_x])
    p_vls[:] = np.nan
    test_cnt = 0
    for i in range(n_x):
        for j in range(i+1, n_x):
            p, t = basic_perm_test(x_0=x[i], x_1=x[j], **test_opts)
            p_vls[i,j] = p
            p_vls[j,i] = p

            test_cnt += 1
            if test_cnt % update_int == 0:
                print('Done with test ' + str(test_cnt) + ' of ' + str(n_total_tests) + '.')

    if not np.isinf(update_int):
        print('Done with all tests.')

    return p_vls
