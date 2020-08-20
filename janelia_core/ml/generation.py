""" Contains tools for generating functions and data for simulation. """

from typing import Callable, Sequence

import numpy as np


class BumpFcnGenerator():
    """
    An object for generating a sequence of functions on the unit hypercube.

    The function f_t is formed as t*f_base, where f_base is a base function
    made up of a sum of Gaussian bumps.

    Bump centers will be uniformly distributed throughout the hypercube.  Bump
    magnitudes and shapes (how fast they fall off in different directions) will
    be randomly generated according to parameters set by the user.

    """

    def __init__(self, d: int = 2, n_bumps: int = 100,
                 bump_peak_vl_range: Sequence[int] = [-1, 1],
                 cov_ev_range: Sequence[int] = [.5, 1]):
        """
        Creates a new BumpFcnGenerator instance.

        Args:
            d: The input dimensionality.

            n_bumps: The number of Gaussian bumps in the base function.

            bump_peak_vl_range: Values for the peak of each bump function will be pulled
            uniformly from this interval.

            cov_ev_range: Values for the eigenvalues of the covariance matrix of the
            Gaussian bump functions will be pulled uniformly from this range.

        """

        self.d = d
        self.n_bumps = n_bumps

        self.ctrs = np.random.uniform(size=[n_bumps, d])
        self.peak_vls = np.random.uniform(low=bump_peak_vl_range[0],
                                          high=bump_peak_vl_range[1],
                                          size=n_bumps)

        # Generate the covariance for each bump function
        eig_vls = np.random.uniform(low=cov_ev_range[0],
                                    high=cov_ev_range[1],
                                    size=[n_bumps, d])

        inv_covs = np.zeros([d, d, n_bumps])
        for b_i in range(n_bumps):
            # Generate a random orthogonal matrix
            u_i = np.linalg.svd(np.random.randn(d, d))[0]
            eig_vls_i = np.diag(1 / eig_vls[b_i, :])
            inv_covs[:, :, b_i] = np.matmul(np.matmul(u_i, eig_vls_i), u_i.transpose())

        self.inv_covs = inv_covs

    def _f(self, x: np.ndarray) -> np.ndarray:
        """ Computes output given input.

        Args:
            x: input of shape n_smps*d_in.

        Returns:
            y: output of shape n_smps
        """

        n_smps = x.shape[0]

        y = np.zeros(n_smps)

        for b_i in range(self.n_bumps):
            x_ctr = x - self.ctrs[b_i, :]

            inv_cov_i = np.squeeze(self.inv_covs[:, :, b_i])

            temp = np.squeeze(np.sum(np.matmul(x_ctr, inv_cov_i.transpose()) * x_ctr, 1))

            y += self.peak_vls[b_i] * np.exp(-1 * temp)

        return y

    def generate(self, scale: float = 1.0) -> Callable:
        """ Generates a function at a given scale.

        Args:
            f: The scale to generate the function at
        """

        def s_f(x):
            return scale * self._f(x)

        return s_f