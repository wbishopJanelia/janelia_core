""" Tools for working with bootstrap analyses. """

import numpy as np


class OnlineNonZeroPVlCalculator():
    """ An object for calculating p-values for non-zero values from streaming bootstrap results.

    This object is designed for use when you want to invert a bootstrap distribution to calculate the p-value
    that a quantity is significantly different from 0, but you don't have the memory available to first perform all the
    bootstrap samples, keeping track of results and then calculating p-values.  In these cases, this object can be
    used to calculate the p-values, by providing it with the results of each bootstrap sample sequentially.

    This object can calculate p-values for multiple variables at once.  The way it should be used is:

        1) Create a new OnlineNonZeroPVlCalculator object, telling it how many variables you will be calculating
        p-values for

        2) Call injest on one or more samples of data; this can be done sequentially

        3) Call p_vls to get p-values calculating after injesting all the samples to date

    Note: This code calculate p-values for a two-sided hypothesis test.

    """

    def __init__(self, n_vars: int):
        """ Creates a new OnlineNonZeroPVlCalculator object.

        Args:
            n_vars: The number of variables we will be calculating p-values for.
        """

        self.n_vars = n_vars
        self.lt0 = np.zeros(n_vars, dtype=np.bool) # Array to keep track of if we've seen values less than 0
        self.gt0 = np.zeros(n_vars, dtype=np.bool) # Array to keep track of if we've seen values greater than 0
        self.nlt0 = np.zeros(n_vars, dtype=np.long) # Array to count how many values are less than 0
        self.ngt0 = np.zeros(n_vars, dtype=np.long) # Array to count how many values are greater than 0
        self.n_smps = 0 # To keep track of how many total samples we've seen

    def injest(self, smp: np.ndarray):
        """ Takes in a new sample of data, which will update calculated p-values.

        Args:

            smp: The samples to injest.  More than one sample can be injested at a time.
            This should be of shape [n_smps, n_vars].

        Raises:
            ValueError: If smp is not a 2-d array
        """

        if smp.ndim != 2:
            raise(ValueError('smp must be a 2-d array of shape [n_smps, n_vars]'))

        # See if there are any values less than or greater than 0 in the sample for each variable
        smp_lt0 = np.any(smp < 0, axis=0)
        smp_gt0 = np.any(smp > 0, axis=0)

        # Now we record if we've ever seen values less than or greater than 0 for each variable
        self.lt0 = np.logical_or(self.lt0, smp_lt0)
        self.gt0 = np.logical_or(self.gt0, smp_gt0)

        # See how many samples are greater than or equal or less than or equal to 0 for each variable
        smp_nlt0 = np.sum(smp <= 0, axis=0)
        smp_ngt0 = np.sum(smp >= 0, axis=0)

        # Now we update the count of the total number of samples greater than or equal to 0 and less than or equal to
        # 0 for each variable

        self.nlt0 = self.nlt0 + smp_nlt0
        self.ngt0 = self.ngt0 + smp_ngt0

        # Now we update count of how many total samples we've seen
        self.n_smps += smp.shape[0]

    def p_vls(self):
        """ Calculates p-values given data injested to date.

        Returns:
            p_vls: The calculated p-values for each variable, will be an array of length n_vars.
        """

        lt0p = (self.nlt0 + self.gt0)/self.n_smps
        gt0p = (self.ngt0 + self.lt0)/self.n_smps

        return 2*np.min(np.stack([lt0p, gt0p]), axis=0)

