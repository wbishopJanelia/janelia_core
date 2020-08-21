""" Contains wandering modules and tools for working with them.

Wandering modules are Torch modules which allow for modifications of their parameters and the gradients for their
parameters during gradient fitting routines.  They are called wandering because in many cases, small amounts of noise
can be added to the the gradients, so these parameters display a random walk when gradients are zero.

"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class WanderingModule(ABC, torch.nn.Module):
    """ Base class for wandering modules.  """

    @abstractmethod
    def bound(self):
        """ Bounds parameters in the module.

        This method takes no arguments, other than self, as the bounds themselves are expected to be saved as attributes
        of the module.
        """
        pass

    @abstractmethod
    def pert_grads(self):
        """ Perturbs (or adjusts) gradients of parameters of the module.

        This method takes no arguments, other than self, as any customization of how gradients are perturbed should
        be controlled through adjusting attributes of the module.  For example, the amount of noise added to gradients
        could be determined by an attribute.
        """
        pass


class BumpFcn(WanderingModule):
    """ A function representing a single exponential bump.

    The bump function is of the form:

        y = m*exp(-sum(((x - c)/w)**2)),

    where x is input, c is the center of the bump, w determines the width of the bump and mm determines the magnitude of
    the bump. Note that x can be multi-dimensional, in which case c and w are vectors of the appropriate dimension.

    """

    def __init__(self, c: torch.Tensor, w: torch.Tensor, m: float, c_bounds: torch.Tensor = None,
                 w_bounds: torch.Tensor = None, c_grad_std: float = 0.0, w_grad_std: float = 0.0,
                 m_grad_gain: float = 0.0, support_p: float = .01):
        """ Creates a new BumpFcn module.

        Args:
            c: The initial center

            w: The initial weight

            m: The initial magnitude

            c_bounds: Bounds for the center parameter.  First row is lower bounds; second row is upper bounds. If
            None, the center will not be bounded.

            w_bounds: Bounds for the width parameter.  First row is lower bounds; second row is upper bounds. If
            None, the width parameters will not be bounded.

            c_grad_std: The standard deviation to use when perturbing gradients for the center parameters with random
            noise

            w_grad_std: The standard deviation to use when perturbing gradients for the width parameters with random
            noise

            m_grad_gain: The gain to use when perturbing the gradient for the magnitude.  The gradient is perturbed by
            subtracting the value m*grad_gain from the gradient for m.

            support_p: The percent of max value in any direction where we define the boundary of support.  This is
            used for skipping function evaluation for input values that are outside of the support.
        """

        super().__init__()

        self.c = torch.nn.Parameter(c)
        self.m = torch.nn.Parameter(torch.Tensor([m]))
        self.w = torch.nn.Parameter(w)
        self.register_buffer('c_bounds', c_bounds)
        self.register_buffer('w_bounds', w_bounds)
        self.c_grad_std = c_grad_std
        self.w_grad_std = w_grad_std
        self.m_grad_gain = m_grad_gain
        self.support_p = support_p
        self.support_k = np.sqrt(-1*np.log(support_p))
        self.n_dims = len(c)

    def forward(self, x: torch.Tensor, small_output: bool = False) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*d_in

            small_output: If false, then output is a tensor of length n_smps, where y[i] is the output for x[i,:]. If
            true, output is a tuple.  The first entry in the tuple is a tuple of indices of points where output is
            non-zero and the second is a tuple of values for these points.

        Returns:
            y: Output of shame n_smps
        """

        n_smps = x.shape[0]

        # Find points in support
        min_bounds = -1 * self.support_k * self.w + self.c
        max_bounds = self.support_k * self.w + self.c
        lower_bound_inds = x > min_bounds
        upper_bound_inds = x < max_bounds
        calc_pts = (torch.all(lower_bound_inds & upper_bound_inds, dim=1)).nonzero().squeeze(dim=1)

        # Make sure we always calculate output for at least 1 input - this ensures gradients are set
        if calc_pts.nelement() == 0:
            calc_pts = torch.tensor([0], dtype=torch.long)

        if not small_output:
            y = torch.zeros(n_smps, device=x.device)
            y[calc_pts] = self.m * torch.exp(
                -1 * torch.sum(((x[calc_pts, :] - self.c) / self.w) ** 2, dim=1))
            return y
        else:
            return [calc_pts,
                    self.m * torch.exp(-1 * torch.sum(((x[calc_pts, :] - self.c) / self.w) ** 2, dim=1))]

    def bound(self):
        """ Bounds parameters of the module. """
        if self.c_bounds is not None:
            small_inds = self.c < self.c_bounds[0]
            big_inds = self.c > self.c_bounds[1]
            self.c.data[small_inds] = self.c_bounds[0, small_inds]
            self.c.data[big_inds] = self.c_bounds[1, big_inds]

        if self.w_bounds is not None:
            small_inds = self.w < self.w_bounds[0]
            big_inds = self.w > self.w_bounds[1]
            self.w.data[small_inds] = self.w_bounds[0, small_inds]
            self.w.data[big_inds] = self.w_bounds[1, big_inds]

    def pert_grads(self):
        """ Perturbs gradients.  """

        if self.c_grad_std != 0:
            self.c._grad += self.c_grad_std*torch.randn(self.n_dims, device=self.c.device)

        if self.w_grad_std != 0:
            self.w._grad += self.w_grad_std*torch.randn(self.n_dims, device=self.c.device)

        if self.m_grad_gain != 0:
            self.m._grad -= self.m_grad_gain*self.m


class SumOfBumpFcns(WanderingModule):
    """ A sum of bump functions, each with learnable centers, widths and magnitudes.

    (See BumpFcn for the functional form of a single bump function).

    """

    def __init__(self, c: torch.Tensor, w: torch.Tensor, m: torch.Tensor,
                 c_bounds: torch.Tensor, w_bounds: torch.Tensor):
        """ Creates a new SumOfBumpFcns modules.

        Currently this module allows for bounding of centers and widths, but does not perturb gradients (which is to say
        the pert_grads() method does nothing).

        Args:

            c: Initial centers.  Each column is a center for a bump.

            w: Initial widths.  Each column is the width parameters for a bump.

            m: Initial magnitudes.  Each entry is the magnitude for a bump.

            c_bounds: Bounds for the center parameter.  First column is lower bounds; second column is upper bounds. If
            None, the center will not be bounded.

            w_bounds: Bounds for the width parameter.  First column is lower bounds; second column is upper bounds. If
            None, the width parameters will not be bounded.

        """

        super().__init__()

        self.c = torch.nn.Parameter(c)
        self.m = torch.nn.Parameter(m)
        self.w = torch.nn.Parameter(w)
        self.register_buffer('c_bounds', c_bounds)
        self.register_buffer('w_bounds', w_bounds)
        self.n_bumps = c.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input from output.

        Args:

            x: Input of shape n_smps*n_dims

        Returns:
             y: Output of length n_smps
        """

        x = x.unsqueeze(-1).expand([-1, -1, self.n_bumps])
        dist_sq = torch.sum(((x - self.c.unsqueeze(0))/self.w.unsqueeze(0))**2, dim=1)
        return torch.sum(self.m*torch.exp(-1*dist_sq), dim=1)

    def bound(self):

        if self.c_bounds is not None:
            expanded_small_c_bounds = self.c_bounds[:, 0:1].expand([-1, self.n_bumps])
            expanded_big_c_bounds = self.c_bounds[:, 1:2].expand([-1, self.n_bumps])

            small_inds = self.c < self.c_bounds[:, 0:1]
            big_inds = self.c > self.c_bounds[:, 1:2]
            self.c.data[small_inds] = expanded_small_c_bounds[small_inds]
            self.c.data[big_inds] = expanded_big_c_bounds[big_inds]

        if self.w_bounds is not None:
            expanded_small_w_bounds = self.w_bounds[:, 0:1].expand([-1, self.n_bumps])
            expanded_big_w_bounds = self.w_bounds[:, 1:2].expand([-1, self.n_bumps])

            small_inds = self.w < self.w_bounds[:, 0:1]
            big_inds = self.w > self.w_bounds[:, 1:2]
            self.w.data[small_inds] = expanded_small_w_bounds[small_inds]
            self.w.data[big_inds] = expanded_big_w_bounds[big_inds]

    def pert_grads(self):
        """ Currently we don't implement any gradient perturbations."""
        pass


class SlowSumOfBumpFcns(WanderingModule):
    """ Implements a sum of bump functions which will be slower in computation on GPU, but may be have memory benefits.

    (See BumpFcn for the functional form of a single bump function).

    """

    def __init__(self, c: torch.Tensor, w: torch.Tensor, m: torch.Tensor,
                 c_bounds: torch.Tensor, w_bounds: torch.Tensor, c_grad_std: float = 0.0, w_grad_std: float = 0.0,
                 m_grad_gain: float = 0.0, support_p: float = .01):
        """ Creates a new SumOfBumpFcns modules.

        This module allows for bounding of centers and widths and perturbing of gradients.

        Args:

            c: Initial centers.  Each column is a center for a bump.

            w: Initial widths.  Each column is the width parameters for a bump.

            m: Initial magnitudes.  Each entry is the magnitude for a bump.

            c_bounds: Bounds for the center parameter.  First column is lower bounds; second column is upper bounds. If
            None, the center will not be bounded.

            w_bounds: Bounds for the width parameter.  First column is lower bounds; second column is upper bounds. If
            None, the width parameters will not be bounded.

            c_grad_std: The standard deviation to use when perturbing gradients for the center parameters with random
            noise

            w_grad_std: The standard deviation to use when perturbing gradients for the width parameters with random
            noise

            m_grad_gain: The gain to use when perturbing the gradient for the magnitude.  The gradient is perturbed by
            subtracting the value m*grad_gain from the gradient for m.

            support_p: The percent of max value in any direction where we define the boundary of support.  This is
            used for skipping function evaluation for input values that are outside of the support.
        """

        super().__init__()

        n_bumps = c.shape[1]
        self.n_bumps = n_bumps

        bump_fcns = [BumpFcn(c=c[:, b_i], w=w[:, b_i], m=m[b_i], c_bounds=c_bounds.t(),
                             w_bounds=w_bounds.t(), c_grad_std=c_grad_std, w_grad_std=w_grad_std,
                             m_grad_gain=m_grad_gain, support_p=support_p) for b_i in range(n_bumps)]

        self.bump_fcns = torch.nn.ModuleList(bump_fcns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: input of shape n_smps*d_in

        Returns:
            y: output of length n_smps
        """

        n_smps = x.shape[0]
        y = torch.zeros(n_smps, device=x.device)
        for b_f in self.bump_fcns:
            y += b_f(x)

        return y

    def bound(self):
        for b_f in self.bump_fcns:
            b_f.bound()

    def pert_grads(self):
        for b_f in self.bump_fcns:
            b_f.pert_grads()

