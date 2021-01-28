""" Contains common objects and functions for developing sparse networks. """

from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class BumpFcn(torch.nn.Module):

    def __init__(self, ctr: torch.Tensor, band_widths: torch.Tensor, mag: float, support_p: float = .01):
        """ Creates a bump function.

        We use exponential functions of the form y(x) = mag*exp(-d), where
            d = sum[((x - ctr)**2)/band_widths]

        Args:

            ctr: The initial center

            band_widths: The initial band widths

            mag: The initial magnitude

            support_p: The percent of max value in any direction where we define the boundary of support

        """
        super().__init__()

        self.ctr = torch.nn.Parameter(ctr)
        self.band_widths = torch.nn.Parameter(band_widths)
        self.mag = torch.nn.Parameter(torch.Tensor([mag]))
        self.support_p = support_p
        self.support_k = np.sqrt(-1*np.log(support_p))

    def forward(self, x: torch.Tensor, small_output: bool = False) -> Union[torch.Tensor, list]:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*d_in

            small_output: If true, output is returned in "small" format.

        Returns:
            y: If small_output is false, this is the output of shape n_smps.  If small_output is true,
            this is a list.  The first entry contains the indices of points output was computed for.  The second
            entry contains the values for these points.  If points are not included in the output, there evaluated
            value is 0.
        """

        n_smps = x.shape[0]

        # Find points in support
        min_bounds = -1*self.support_k*self.band_widths + self.ctr
        max_bounds = self.support_k*self.band_widths + self.ctr
        lower_bound_inds = x > min_bounds
        upper_bound_inds = x < max_bounds
        calc_pts = (torch.all(lower_bound_inds & upper_bound_inds, dim=1)).nonzero().squeeze(dim=1)

        # Make sure we always calculate output for at least 1 input - this ensures gradients are set
        if calc_pts.nelement() == 0:
            calc_pts = torch.tensor([0], dtype=torch.long)

        if not small_output:
            y = torch.zeros(n_smps, device=x.device)
            y[calc_pts] = self.mag*torch.exp(-1*torch.sum(((x[calc_pts,:] - self.ctr)/self.band_widths)**2, dim=1))
            return y
        else:
            return [calc_pts, self.mag*torch.exp(-1*torch.sum(((x[calc_pts,:] - self.ctr)/self.band_widths)**2, dim=1))]

    def bound(self, ctr_bounds: Sequence = [0, 1],
              bandwidth_bounds: Sequence = [.01, 10]):
        """  Applies bounds to the centers and bandwidths.

        Bounds are applied element-wise.

        Args:

            ctr_bounds: The bounds to force centers to be between. If None, no bounds are enforced.

            bandwidth_bounds: The bounds to force band widths to be between. If None, no bounds are enforced.

        """

        if ctr_bounds is not None:
            small_inds = self.ctr < ctr_bounds[0]
            big_inds = self.ctr > ctr_bounds[1]
            self.ctr.data[small_inds] = ctr_bounds[0]
            self.ctr.data[big_inds] = ctr_bounds[1]

        if bandwidth_bounds is not None:
            small_inds = self.band_widths < bandwidth_bounds[0]
            big_inds = self.band_widths > bandwidth_bounds[1]
            self.band_widths.data[small_inds] = bandwidth_bounds[0]
            self.band_widths.data[big_inds] = bandwidth_bounds[1]

    def add_noise_to_grads(self, center_k: float = .01, mag_k: float = .01,
                           bandwidth_k: float = .01):
        """ Adds noise to calculated gradients.

        This is an experimental feature and has not been found to be actually helpful in practice.

        Args:

            center_k: The standard deviation of noise to add to gradients for center parameters.

            mag_k: The amount to grow the magnitude by.  (This is not random)

            bandwidth_k: The standard deviation of the noise to add to the band widths.
        """

        if center_k != 0:
            ctr_grad_noise = torch.randn(2, device=self.ctr.device)
            self.ctr._grad += center_k*ctr_grad_noise

        if mag_k != 0:
            self.mag._grad += mag_k*self.mag

        if bandwidth_k != 0:
            bound_grad_noise = torch.randn(2, device=self.band_widths.device)
            self.band_widths._grad += bandwidth_k*bound_grad_noise


class CenteredBumpFcn(torch.nn.Module):

    def __init__(self, band_widths, mag):
        """ Creates a bump function, with a center fixed at 0.

         We use exponential functions of the form y(x) = mag*exp(-d), where
             d = sum[(x**2)/band_widths]

         Args:

             band_widths: The initial band widths

             mag: The initial magnitude

         """

        super().__init__()

        self.band_widths = torch.nn.Parameter(band_widths)
        self.mag = torch.nn.Parameter(torch.Tensor([mag]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*d_in


        Returns:
            y: This is the output of shape n_smps.
        """

        return self.mag*torch.exp(-1*torch.sum((x/self.band_widths)**2, dim=1))

    def bound(self, bandwidth_bounds: Sequence = [.01, 10]):
        """  Applies bounds to bandwidths.

        Bounds are applied element-wise.

        Args:

            bandwidth_bounds: The bounds to force band widths to be between. If None, no bounds are enforced.

        """

        if bandwidth_bounds is not None:
            small_inds = self.band_widths < bandwidth_bounds[0]
            big_inds = self.band_widths > bandwidth_bounds[1]
            self.band_widths.data[small_inds] = bandwidth_bounds[0]
            self.band_widths.data[big_inds] = bandwidth_bounds[1]


class FunctionSequenceGenerator():
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
        Creates a new F_t object.

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


def generate_pts(d: int = 2, n_smps_per_dim: int = 100):
    """ Generates points on a hyper grid for evaluating a function.

    Args:
        d: The dimensionality of the grid.

        n_smps_per_dim: The number of samples per dimension to generate.
    """

    dim_coords = (1/n_smps_per_dim)*np.arange(n_smps_per_dim)
    all_coords = [dim_coords]*d
    grid_coords = np.meshgrid(*all_coords)
    coords = np.stack([np.ravel(g) for g in grid_coords]).transpose()

    return coords


class SumOfBumpFcns(torch.nn.Module):

    def __init__(self, n_bumps: int, ctr_range: np.ndarray = None, bandwidth_range: np.ndarray = None,
                 mag_range: np.ndarray = None, support_p: float = .01, d: int = None):
        """ Creates a new SumOfBumpFcns module.

        Args:

            ctr_range: Range to randomly create centers in.  ctr_range[i,:] gives the range for dimension i.
            If None, will be [0, 1.0] for all dimensions.

            bandwidth_range: Range to create bandwidths in.  bandwidth_range[i,:] gives the range for dimension i.
            If None, will be [.1, 1] for all dimensions.

            mag_range: Range to create magnitudes in.  If None, will be [-.1 ,.1].

            support_p: The value of support_p for each bump function.

            d: The number of dimensions of input.  This is only consulted if ctr_range or bandwidth_range is None.

        """
        super().__init__()

        if ctr_range is None:
            ctr_range = np.tile(np.asarray([[0, 1.0]]), (d, 1))

        if bandwidth_range is None:
            bandwidth_range = np.tile(np.asarray([[.1, 1]]), (d, 1))

        if mag_range is None:
            mag_range = np.asarray([-.1, .1])

        # Convert range parameters to torch tensors.  Here we set the data type of these tensors to the torch default.
        ctr_range = torch.tensor(ctr_range, dtype=torch.get_default_dtype())
        bandwidth_range = torch.tensor(bandwidth_range, dtype=torch.get_default_dtype())
        mag_range = torch.tensor(mag_range, dtype=torch.get_default_dtype())

        n_dims = ctr_range.shape[0]
        ctr_w = ctr_range[:, 1] - ctr_range[:, 0]
        bandwidth_w = bandwidth_range[:, 1] - bandwidth_range[:, 0]
        mag_w = mag_range[1] - mag_range[0]

        bump_fcns = [BumpFcn(ctr=torch.rand(n_dims) * ctr_w + ctr_range[:, 0],
                             band_widths=torch.rand(n_dims) * bandwidth_w + bandwidth_range[:, 0],
                             mag=torch.rand(1) * mag_w + mag_range[0],
                             support_p=support_p) for b_i in range(n_bumps)]

        self.bump_fcns = torch.nn.ModuleList(bump_fcns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            y: Output of shape n_smps
        """

        # TODO: Can we evaluate all the bump functions in parallel on the GPU?
        return torch.sum(torch.stack([b_f(x) for b_f in self.bump_fcns]), dim=0)

    def bound(self, ctr_bounds: Sequence = [0, 1],
              bandwidth_bounds: Sequence = [.01, .3]):
        """ Applies bounds to each bump function.

        Args:

            ctr_bounds: Bounds to apply to centers.  Bound is applied elemnt-wise.

            bandwidth_bounds: Bounds to apply to band widths.  Bound is applied element-wise.
        """

        for b_f in self.bump_fcns:
            b_f.bound(ctr_bounds=ctr_bounds, bandwidth_bounds=bandwidth_bounds)

    def add_noise_to_grads(self, center_k: float = .01, mag_k: float = .01,
                           bandwidth_k: float = .01):
        """ Adds noise to the gradients of each bump function.

        Note: This is an experimental feature and has not been found to be useful in practice.

        Args:
            center_k, mag_k, bandwidth_k: See documentation in corresponding function in BumpFcn.

        """
        for b_f in self.bump_fcns:
            b_f.add_noise_to_grads(center_k=center_k, mag_k=mag_k, bandwidth_k=bandwidth_k)


def plot_2d_f(f: Callable, n_smps_per_dim=100):
    """ Plots a 2-d function on the unit square.

    Args:
        f: The function to plot

        n_smps_per_dim: The number of samples per dimension to use when generating
        a grid to use for visualizing the function.
    """

    pts = generate_pts(d=2, n_smps_per_dim=n_smps_per_dim)

    y_grid = f(pts).reshape([n_smps_per_dim, n_smps_per_dim])

    plt.imshow(y_grid)
    plt.colorbar()


