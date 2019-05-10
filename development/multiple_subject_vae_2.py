""" Contains function and class definitions for the multiple_subject_vae_2 notebook.

The spirit of the code presented here is to allow us to define functions and objects without
clutering the notebook.  However, since this is development/prototype code, to speed development
it will not be documented to a normal standard.

"""
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch


class BernoulliCondDistribution(torch.nn.Module):
    """ A module for working with conditional Bernoulli distributions.

    """

    def __init__(self, log_prob_fcn: torch.nn.Module):
        """ Creates a BernoulliCondDistribution variable.

        Args:
            n_vars: The number of variables that the Bernoulli random variable is conditioned on.
        """

        super().__init__()
        self.log_prob_fcn = log_prob_fcn

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape nSmps*d, where d is the number variables we
            condition on.

            y: Values we desire the log probability for.  Of shape nSmps*d_y, where d_y is the number of
            random variables in the conditional distribution.

        Returns:
            log_prob: the log probability of each sample

        """

        if len(y.shape) > 1:
            raise(ValueError('y must be a vector'))

        n_smps = x.shape[0]

        zero_inds = y == 0

        log_nz_prob = self.log_prob_fcn(x).reshape([n_smps])

        log_prob = torch.zeros(n_smps)
        log_prob[~zero_inds] = log_nz_prob[~zero_inds]
        log_prob[zero_inds] = torch.log(1 - torch.exp(log_nz_prob[zero_inds]))

        return log_prob

    def sample(self, x: torch.Tensor):
        """ Samples from P(y|x)

        Args:
            x: Data we condition on.  Of shape nSmps*d.

        Returns:
            smp: The sampled data of shape nSmps.
        """

        probs = torch.exp(self.log_prob(x, torch.ones(x.shape[0])))

        bern_dist = torch.distributions.bernoulli.Bernoulli(probs)
        return bern_dist.sample().byte()


class CondGaussianDistribution(torch.nn.Module):
    """ A general object for representing distributions over random variables with Gaussian conditional distributions.
    """

    def __init__(self, mn_fcn: torch.nn.Module, std_fcn: torch.nn.Module):
        """ Creates a CondGaussianDistribution object.

        Args:
            mn_fcn: A function which maps from x, the conditioning input, to the mean of a Gaussian distribution.
            Given an input of shape nSmps*nXDims, it should return an output of nSmps*nMnDims

            std_fcn: A function which maps from x to a vector of standard deviations for a Gaussian distribution.
            Should accept input and return output of the same shape as mn_fcn.
        """

        super().__init__()
        self.mn_fcn = mn_fcn
        self.std_fcn = std_fcn

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape nSmps*d_x, where d_x is the number variables we
            condition on.

            y: Values we desire the log probability for.  Of shape nSmps*d_y, where d_y is the number of
            random variables in the conditional distribution.

        Returns:
            ll: Log-likelihood of each sample.

        """

        d = x.shape[1]

        mn = self.mn_fcn(x)
        std = self.std_fcn(x)

        ll = -.5*torch.sum(((y - mn)/std)**2, 1)
        ll -= .5*torch.sum(2*torch.log(std), 1)
        ll -= .5*d*torch.log(torch.tensor([math.pi]))

        return ll

    def rsample(self, x: torch.Tensor) -> torch.Tensor:
        """ Samples from reparameterized form of P(y|x).

        Args:
            x: Data we condition on.  Of shape nSmps*d, where d is the number variables we
            condition on.

        Returns:
            y: sampled data of shape nSmps*d_out, where d_out is the number of random variables the distribution is
            over.
        """

        mn = self.mn_fcn(x)
        std = self.std_fcn(x)

        n_smps = mn.shape[0]
        d = mn.shape[1]

        z = torch.randn(n_smps, d)

        return mn + z*std


class CondSpikeSlabDistribution(torch.nn.Module):
    """ A general object for working with conditional spike and slab distributions.
    """

    def __init__(self, spike_dist: torch.nn.Module, slab_dist: torch.nn.Module, d: int):
        """ Creates a CondSpikeSlab object.

        Args:
            spike_dist: A module representing a distribution for the probability
            of a coupling having a zero coefficient given some conditioning data.

            slab_dist: A module represnting the distribution over the values of
            non-zero couplings given conditioning data.

            d: The dimensionality of the random variable this is a distriubtion over.

        """

        super().__init__()

        self.spike_dist = spike_dist
        self.slab_dist = slab_dist
        self.d = d

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape nSmps*d_x, where d_x is the number variables we
            condition on.

            y: Values we desire the log probabily for.  Of shape nSmps*d_y, where d_y is the number of
            random variables in the conditional distribution.

        Returns:
            ll: Log-likelihood of each sample.

        """

        y_d = y.shape[1]

        non_zero_inds = torch.sum(y != 0, dim=1) == y_d

        # Log-likelihood due to spike distribution
        y_non_zero = torch.sum(y != 0, dim=1) == y_d
        ll = self.spike_dist.log_prob(x, non_zero_inds)

        # Log-likelihood due to slab distribution
        ll[non_zero_inds] += self.slab_dist.log_prob(x[non_zero_inds, :], y[non_zero_inds, :])

        return ll

    def sample(self, x: torch.Tensor) -> list:
        """ Generates samples given conditioning data.

        Sampling will be done so that gradients can still be passed through any non-zero entries of the sample.

        Args:
            x: Conditioning data of shape nSmps*d, where d is the number of variables we
            condition on.

        Returns:
            support: A vector of length nSmps.  support[i] is 1 if the sample corresponding to x[i,:] is non-zero

            nz_vls: A tensor of non-zero values.  nz_vls[j] contains the value for the j^th non-zero entry in support.
            In other words, nz_vls gives the non-zero values corresponding to the samples in x[support, :].
        """

        support = self.spike_dist.sample(x)
        nz_vls = self.slab_dist.rsample(x[support,:]).squeeze()

        return [support, nz_vls]


class ConstantRealFcn(torch.nn.Module):
    """ Object for representing function which are constant w.r.t to input and take values anywhere in the reals.
     """

    def __init__(self, n_dims: int):
        """ Creates a ConstantRealFcn object.

        Args:
            n_dims: the number of dimensions of the output of the function
        """

        super().__init__()

        self.n_dims = n_dims

        self.vl = torch.nn.Parameter(torch.zeros(n_dims), requires_grad=True)
        torch.nn.init.normal_(self.vl, 0, .1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Conditioning data of shape nSmps*d_in, where d_in is the number of variables we
            condition on.

        Returns:
            y: output of shape nSmps*d_out
        """

        n_smps = x.shape[0]

        return self.vl.unsqueeze(0).expand(n_smps, self.n_dims)


class ConstantLowerBoundedFcn(torch.nn.Module):
    """ Object for representing function which are constant w.r.t to input and take values anywhere in a range [m, \inf)
    """

    def __init__(self, n_dims: int, lower_bound: float = .000000001):
        """ Creates a ConstantLowerBoundedFcn object.

        Args:
            n_dims: the number of dimensions of the output of the function

            min_vl: The minimum value the function can take on.
        """

        super().__init__()

        self.n_dims = n_dims

        self.lower_bound = torch.tensor([lower_bound])

        self.log_vl = torch.nn.Parameter(torch.zeros(n_dims), requires_grad=True)
        torch.nn.init.normal_(self.log_vl, 0, .1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Conditioning data of shape nSmps*d_in, where d_in is the number of variables we
            condition on.

        Returns:
            y: output of shape nSmps*d_out
        """

        n_smps = x.shape[0]

        vl = torch.exp(self.log_vl) + self.lower_bound

        return vl.unsqueeze(0).expand(n_smps, self.n_dims)


class ConstantLowerUpperBoundedFcn(torch.nn.Module):
    """ Object for representing a constant function which can take on output in a given range. """

    def __init__(self, n_dims: int, lower_bound: float = .01, upper_bound: float = 1.0,
                 init_value:float = None):
        """ Creates a ConstantLowerBoundedFcn object.

        Args:
            n_dims: the number of dimensions of the output of the function

            min_vl: The minimum value the function can take on.
        """

        super().__init__()

        self.n_dims = n_dims

        self.lower_bound = torch.tensor([lower_bound])
        self.upper_bound = torch.tensor([upper_bound])

        self.v = torch.nn.Parameter(torch.zeros(n_dims), requires_grad=True)

        if init_value is None:
            init_value = .5*(lower_bound + upper_bound)

        init_v = np.arctanh(2*(init_value - lower_bound)/(upper_bound - lower_bound) - 1)
        self.v.data = torch.tensor(init_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Conditioning data of shape nSmps*d_in, where d_in is the number of variables we
            condition on.

        Returns:
            y: output of shape nSmps*d_out
        """

        n_smps = x.shape[0]

        vl = (.5*torch.tanh(self.v) + .5)*(self.upper_bound - self.lower_bound) + self.lower_bound

        return vl.unsqueeze(0).expand(n_smps, self.n_dims)

class LogBumpFcn(torch.nn.Module):
    """ A module representing a log "bump" function with trainable parameters of the form:

            y = log(g*f(sum((x-c)/s)^2))),

        where $y \in R$ is the output, x \in R^d is the input, c \in R^d is a center position,
        s \in R^d is a "standard deviation" vector which determines how fast the bump falls off
        in each direction and g is a gain \in [0, 1].  Above f() is a non-linearity.

    """

    def __init__(self, n_vars: int, f_type = 'exp'):
        """ Creates a Bump Fcn object.

        Args:

            n_vars: The dimensionality of the domain of the function.
        """

        super().__init__()

        self.f_type = f_type

        self.ctr = torch.nn.Parameter(torch.zeros(n_vars), requires_grad=True)
        torch.nn.init.uniform_(self.ctr, 0, 1)

        # Log "Variances" determining how fast bumps fall off in each direction
        self.ctr_stds = ConstantLowerUpperBoundedFcn(n_vars, lower_bound=.02, upper_bound=10.0,
                                                     init_value=1.0)

        self.log_gain_vl = ConstantLowerUpperBoundedFcn(1, lower_bound=-3.0, upper_bound=0.0,
                                                        init_value=-2.0)

    def forward(self, x:torch.Tensor):
        """ Computes output of function given input.

        Args:
            x: Input of shape nSmps*d

        Returns:
            y: Output of shape nSmps
        """

        place_holder_input = torch.zeros(1)

        ctr_stds = self.ctr_stds(place_holder_input).squeeze()
        log_gain = self.log_gain_vl(place_holder_input).squeeze()


        x_ctr = x - self.ctr
        x_ctr_scaled = x_ctr/ctr_stds
        x_dist = torch.sum(x_ctr_scaled**2, dim=1)

        if self.f_type == 'exp':
            return log_gain + -1*x_dist
        elif self.f_type == 'tanh':
            return log_gain + .5*torch.tanh(x_dist) + .5
        else:
            raise(RuntimeError('The nonlinearity ' + self.f_type + ' is not recogonized.'))


def visualize_spike_slab_distribution(d, x_range = [0, 1], y_range = [0, 1], n_points_per_side = 100,
                                      smp_x=None, smp_y=None):

    grid = np.mgrid[x_range[0]:x_range[1]:n_points_per_side * 1j,
                    y_range[0]:y_range[1]:n_points_per_side * 1j]

    grid_vec = np.stack([np.ravel(grid[0, :, :]), np.ravel(grid[1, :, :])]).transpose()
    grid_vec = torch.from_numpy(grid_vec.astype('float32'))

    # Plot probability of non-zero entries
    plt.figure()
    plt.subplot(2, 2, 1)
    nz_probs = torch.exp(d.spike_dist.log_prob(grid_vec, torch.ones(grid_vec.shape[0]))).detach().numpy()
    nz_probs = np.reshape(nz_probs, [n_points_per_side, n_points_per_side])
    plt.imshow(nz_probs, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('Prob. Non-Zero')

    # Plot mean of spike distribution
    plt.subplot(2, 2, 2)
    slab_mn = d.slab_dist.mn_fcn(grid_vec).detach().numpy()
    slab_mn = np.reshape(slab_mn, [n_points_per_side, n_points_per_side])
    plt.imshow(slab_mn, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('Slab Mean')

    # Plot standard deviation of slab distribution
    plt.subplot(2, 2, 3)
    slab_std = d.slab_dist.std_fcn(grid_vec).detach().numpy()
    slab_std = np.reshape(slab_std, [n_points_per_side, n_points_per_side])
    plt.imshow(slab_std, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('Slab Std')

    # Plot emperical data
    if smp_x is not None:
        a = plt.subplot(2,2,4)
        smp_x = smp_x.detach().numpy()
        smp_y = smp_y.detach().numpy()

        cmap = matplotlib.cm.get_cmap('PiYG')

        max_y = np.max(smp_y)
        min_y = np.min(smp_y)
        max_mag = np.max([max_y, -1*min_y])

        for i in range(smp_x.shape[0]):
            c_vl = smp_y[i]/(2 * max_mag) + .5
            clr = cmap(c_vl)
            p = plt.plot(smp_x[i, 1], smp_x[i, 0], 'ko', alpha=.5, markerfacecolor=clr)

        plt.xlim(x_range)
        plt.ylim(y_range)

        a.set_aspect('equal', 'box')


def visualize_bernoulli_distribution(d, x_range = [0, 1], y_range = [0, 1], n_points_per_side = 100,
                                     smp_x = None, smp_y = None):


    grid = np.mgrid[x_range[0]:x_range[1]:n_points_per_side * 1j,
           y_range[0]:y_range[1]:n_points_per_side * 1j]

    grid_vec = np.stack([np.ravel(grid[0, :, :]), np.ravel(grid[1, :, :])]).transpose()
    grid_vec = torch.from_numpy(grid_vec.astype('float32'))

    # Plot probability of non-zero entries
    plt.figure()
    plt.subplot(1, 2, 1)
    nz_probs = torch.exp(d.log_prob(grid_vec, torch.ones(grid_vec.shape[0]))).detach().numpy()
    nz_probs = np.reshape(nz_probs, [n_points_per_side, n_points_per_side])
    plt.imshow(nz_probs, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('Prob. Non-Zero')

    # Plot emperical data
    if smp_x is not None:
        a = plt.subplot(1,2,2)

        cmap = matplotlib.cm.get_cmap('PiYG')

        max_y = np.max(smp_y)
        min_y = np.min(smp_y)
        max_mag = np.max([max_y, -1*min_y])

        for i in range(smp_x.shape[0]):
            c_vl = smp_y[i]/(2 * max_mag) + .5
            clr = cmap(c_vl)
            p = plt.plot(smp_x[i, 1], smp_x[i, 0], 'ko', alpha=.5, markerfacecolor=clr)

        plt.xlim(x_range)
        plt.ylim(y_range)

        a.set_aspect('equal', 'box')


def make_bernoulli_vi_debug_figure(q, p, smp_x, smp_y, grads, x_range = [0, 1], y_range = [0, 1],
                                   n_points_per_side = 100):


    grid = np.mgrid[x_range[0]:x_range[1]:n_points_per_side * 1j,
           y_range[0]:y_range[1]:n_points_per_side * 1j]

    grid_vec = np.stack([np.ravel(grid[0, :, :]), np.ravel(grid[1, :, :])]).transpose()
    grid_vec = torch.from_numpy(grid_vec.astype('float32'))

    # Plot probability of non-zero entries for p distribution
    plt.figure()
    plt.subplot(1, 4, 1)
    nz_probs = torch.exp(p.log_prob(grid_vec, torch.ones(grid_vec.shape[0]))).detach().numpy()
    nz_probs = np.reshape(nz_probs, [n_points_per_side, n_points_per_side])
    plt.imshow(nz_probs, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('P Prob. Non-Zero')

    # Plot probability of non-zero entries for q distribution
    plt.subplot(1, 4, 2)
    nz_probs = torch.exp(q.log_prob(grid_vec, torch.ones(grid_vec.shape[0]))).detach().numpy()
    nz_probs = np.reshape(nz_probs, [n_points_per_side, n_points_per_side])
    plt.imshow(nz_probs, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    plt.colorbar()
    plt.title('Q Prob. Non-Zero')

    # Plot samples
    a = plt.subplot(1,4,3)
    for i in range(smp_x.shape[0]):
        if smp_y[i] == 0:
            clr = [0.0, 0.0, 0]
        else:
            clr = [1.0, 0, 0]

        p = plt.plot(smp_x[i, 1], smp_x[i, 0], 'o', markerfacecolor=clr)

    plt.xlim(x_range)
    plt.ylim(y_range)
    a.set_aspect('equal', 'box')

    # Plot gradients
    grad_mags = np.sqrt(np.sum(grads**2,1))
    max_mag = np.max(grad_mags)
    grads = .1*grads/max_mag

    a = plt.subplot(1,4,4)
    for i in range(smp_x.shape[0]):
        plt.plot([smp_x[i, 1], smp_x[i, 1] + grads[i,1]],
                 [smp_x[i, 0], smp_x[i, 0] + grads[i,0]],
                 'r-')
        plt.plot(smp_x[i,1], smp_x[i,0], 'ko', markersize=.5)

    grad_sum = np.sum(grads, 0)
    plt.plot([.5, .5+grad_sum[1]], [.5, .5+grad_sum[0]], 'g-')

    plt.xlim(x_range)
    plt.ylim(y_range)
    a.set_aspect('equal', 'box')


