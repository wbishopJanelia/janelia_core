""" Contains basic torch modules, supplementing those natviely in Torch.
"""

from typing import Sequence

import numpy as np
import torch
from torch.nn.functional import relu

from janelia_core.math.basic_functions import int_to_arb_base

class Bias(torch.nn.ModuleList):
    """ Applies a bias transformation to the data y = x + o """

    def __init__(self, d: int, init_std: float = .1):
        """ Creates a Bias object.

        Args:
            d: The dimensionality of the input and output

            init_std: The standard deviation of the normal distribution initial biases are pulled from.
        """

        super().__init__()
        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, std=init_std)
        self.register_parameter('o', o)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return x + self.o


class ConstantBoundedFcn(torch.nn.Module):
    """ Object for representing a constant function which can produce output in a bounded range. """

    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray, init_value: np.ndarray = None):
        """ Creates a ConstantLowerBoundedFcn object.

        Args:
            lower_bound, upper_bound: the lower and upper bounds the output of the function can take on.  These
            should be arrays providing the bounds for each dimension of output.

            init_value: If provided, this is the constant output the function is initialized to.  Should be an
            array providing initial values for each dimension. If not provided, the constant value will be initialized
            to be halfway between the lower and upper bound.
        """

        super().__init__()

        n_dims = len(lower_bound)
        self.n_dims = n_dims

        self.register_buffer('lower_bound', torch.tensor(lower_bound).float())
        self.register_buffer('upper_bound', torch.tensor(upper_bound).float())

        self.v = torch.nn.Parameter(torch.zeros(n_dims), requires_grad=True)

        if init_value is None:
            init_value = .5*(lower_bound + upper_bound)

        init_v = np.arctanh(2*(init_value - lower_bound)/(upper_bound - lower_bound) - 1)
        self.v.data = torch.tensor(init_v)
        self.v.data = self.v.data.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Input data of shape n_smps*d_in.

        Returns:
            y: output of shape nSmps*d_out

        """

        n_smps = x.shape[0]

        vl = (.5*torch.tanh(self.v) + .5)*(self.upper_bound - self.lower_bound) + self.lower_bound

        return vl.unsqueeze(0).expand(n_smps, self.n_dims)


class DenseLayer(torch.nn.Module):
    """ A layer which concatenates its input to it's output. """

    def __init__(self, m: torch.nn.Module):
        """ Creates a DenseLayer object.

        Args:

            m: The module which input is passed through.  The output of this module is concatenated to
            the input to form the final output of the module.
        """

        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input from output. """

        return torch.cat((x, self.m(x)), dim=1)


class DenseLNLNet(torch.nn.Module):
    """ A network of densely connected linear, non-linear units. """

    def __init__(self, nl_class: type, d_in: int, n_layers: int, growth_rate: int, bias: bool = False):
        """ Creates a DenseLNLNet object.

        Args:
              nl_class: The class to construct the non-linear activation functions from, e.g., torch.nn.ReLU

              d_in: Input dimensionality to the network

              n_layers: The number of layers in the network.

              growth_rate: The number of unique features computed by each layer.  The output dimensionality of
              the network will be: d_in + n_layers*growth_rate.

              bias: True if linear layers should have a bias.

        """

        super().__init__()

        for i in range(n_layers):

            linear_layer = torch.nn.Linear(in_features=d_in + i*growth_rate, out_features=growth_rate, bias=bias)

            dense_layer = DenseLayer(torch.nn.Sequential(linear_layer, nl_class()))

            self.add_module('dense_lnl_' + str(i), dense_layer) # Add linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input given output. """

        for module in self._modules.values():
            x = module(x)

        return x


class SumOfTiledHyperCubeBasisFcns(torch.nn.Module):
    """ A module to represent a function which is a sum of tiled hypercube basis functions.

        The hypercubes tile, in an overlapping fashion, a volume.  To specify a layout of hypercubes the user:

            (1) Specifies the range of each dimension that should be covered

            (2) Specifies a number of divisions to break the range of each dimension into (see illustration below). These
            divisions *do not* directly correspond to hypercubes.  (See 3)

            (3) Specifies how many divisions make up the side of a hypercube in each dimension.  For non-overlapping
            hypercubes 1 division makes up the side of 1 hypercube.  Increasing the number of divisions per side of each
            hypercube results in overlapping hypercubes (see illustration below).

            (4) Final hypercubes are constructed to respect the hypercube sides set for each dimension.  Each hypercube
            has it's own learnable magnitude.

            Example of breaking up a dimension into divsions and overlapping hypercube sides with 2 divisions per
            hypercube side:

                |-|-|-|-|-|-|-|-| : Each block a division (e.g., 8 divisions)
                ^               ^
                |               |
                start_range     end_range
                |               |
                |               |
                |               |
                |               |
              |- -|             |   : (Notice padding so that first and last hypercubes run over the valid range)
                |- -|           |
                  |- -|         |
                       ...      |
                              |- -|


         Note: This object has been optimized for speed.  Specifically, by having hypercubes defined with respect to
         a base set of divisions, it is possible to take an input point and use an efficient hashing function to
         determine all hypercubes that it falls in.  More ever, by including padding of the hypercubes, we ensure
         that each input point to the function anywhere in the user specified range falls within the *same* number of
         hypercubes.  These two things make forward evaluation of the function efficient.
     """

    def __init__(self, n_divisions_per_dim: Sequence[int], dim_ranges: np.ndarray, n_div_per_hc_side_per_dim: np.ndarray):
        """
        Creates a SumOfTiledHyperCubeBasisFcns object.

        Args:

            n_divisions_per_dim: n_divisions_per_dim[i] gives the number of divisions for dimension i.

            dim_ranges: The range for dimension i is dim_ranges[i,0] <= x[i] < dim_ranges[i,1]

            n_div_per_hc_side_per_dim: The number of divisions per hypercube side for each dimension

        """

        super().__init__()

        n_dims = dim_ranges.shape[0]
        self.register_buffer('n_dims', torch.Tensor([n_dims]))

        div_widths = (dim_ranges[:, 1] - dim_ranges[:, 0])/n_divisions_per_dim
        self.register_buffer('div_widths', torch.Tensor(div_widths))

        self.register_buffer('min_dim_ranges', torch.Tensor(dim_ranges[:, 0]))
        self.register_buffer('max_dim_ranges', torch.Tensor(dim_ranges[:, 1]))

        # Determine the order of dimensions for the purposes of linearalization - we want the dimension
        # which will have the most active bump functions for a given point to be last.  This will allow us
        # to specify the largest contiguous chunks of the array holding bump function magnitudes.
        dim_order = np.argsort(n_div_per_hc_side_per_dim)
        self.register_buffer('dim_order', torch.Tensor(dim_order).long())

        # Determine how many bump functions per dimension there are - we order this according to dim_order
        n_bump_fcns_per_dim = np.asarray([n_div + n_div_per_block - 1 for n_div, n_div_per_block in
                                          zip(n_divisions_per_dim, n_div_per_hc_side_per_dim)])
        n_bump_fcns_per_dim = n_bump_fcns_per_dim[dim_order]

        # Order n_div_per_hs_side_per_dim according to dim_order too - the rest of the code
        # in this function will assume this order
        n_div_per_hc_side_per_dim = n_div_per_hc_side_per_dim[dim_order]

        # Pre-calculate factors we need for linearalization - saved in order according to dim_order
        dim_factors = np.ones(n_dims)
        for d_i in range(n_dims-2, -1, -1):
            dim_factors[d_i] = dim_factors[d_i + 1]*n_bump_fcns_per_dim[d_i + 1]
        self.register_buffer('dim_factors', torch.Tensor(dim_factors).long())

        # Calculate offset vector for looking up active bump functions for each point.  This offset vector
        # can be added to the linear index of the first active bump function for a point to get the indices of
        # all active bump functions for that point

        n_active_bump_fcns = (np.cumprod(n_div_per_hc_side_per_dim)[-1]).astype('long')

        if n_dims > 1:
            n_minor_dim_repeats = np.cumprod(n_div_per_hc_side_per_dim[0:-1])[-1]
        else:
            n_minor_dim_repeats = 1


        bump_ind_offsets = torch.arange(n_div_per_hc_side_per_dim[-1]).repeat(n_minor_dim_repeats).long()
        cur_chunk_size = 1
        for d_i in range(n_dims-2, -1, -1):
            cur_chunk_size = cur_chunk_size * n_div_per_hc_side_per_dim[d_i + 1]
            cur_n_chunks = int(n_active_bump_fcns/cur_chunk_size)
            cur_n_stacked_chunks = n_div_per_hc_side_per_dim[d_i]
            for c_i in range(cur_n_chunks):
                cur_chunk_start_ind = c_i*cur_chunk_size
                cur_chunk_end_ind = cur_chunk_start_ind + cur_chunk_size
                mod_i = c_i % cur_n_stacked_chunks
                bump_ind_offsets[cur_chunk_start_ind:cur_chunk_end_ind] += dim_factors[d_i]*mod_i

        self.register_buffer('bump_ind_offsets', bump_ind_offsets)

        # Initialize the magnitudes of each bump function.  We initialize to zero so that if there is never
        # any training data that falls within the support of a bump, that bump will have a zero magnitude.
        # Also, we put all magnitudes in a single 1-d vector for fast indexing

        n_bump_fcns = np.cumprod(n_bump_fcns_per_dim)[-1]
        self.b_m = torch.nn.Parameter(torch.zeros(n_bump_fcns), requires_grad=True)

    def _x_to_idx(self, x: torch.Tensor, run_checks: bool = True):
        """ Given x data computes the indices of active bump functions for each point.

        Args:
            x: Input data of shape n_smps*d_x

            run_checks: True if input should be checked for expected properties

        Returns:
            idx: Indices of active bump functions for each point.  Of shape n_smps*n_active,
            where n_active is the number of active bump functions for each point.

        Raises:
            ValueError: If check_range is true and one or more x values are not in the valid range for the function.
        """

        n_smps = x.shape[0]
        n_x_dims = x.shape[1]

        if run_checks:
            if n_x_dims != self.n_dims:
                raise(ValueError('x does not have expected number of dimensions.'))
            if torch.any(x < self.min_dim_ranges) | torch.any(x >= self.max_dim_ranges):
                raise(ValueError('One or more x values falls outside of the valid range for the function.'))

        # Determine the division along each dimension each point falls into
        dim_div_inds = torch.floor((x - self.min_dim_ranges)/self.div_widths).long()

        # Sort dimensions in encoding order
        dim_div_inds = dim_div_inds[:, self.dim_order]

        # Determine the first function that is active for each point in each dimension.
        # We define bin indices so that the index of the first bin that is active in a dimension is equal to the
        # division index.
        dim_first_bin_inds = torch.sum(dim_div_inds*self.dim_factors, dim=1).view([n_smps, 1])

        return dim_first_bin_inds + self.bump_ind_offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes input given output.

        Args:
            x: Input of shape n_smps*d_x.  Each x point should be within the region specified when creating the
            SumOfTiledHyperCubeBasisFcns object.

        Returns:
            y: Output of shape n_smps*1.
        """
        n_smps = x.shape[0]
        return torch.sum(self.b_m[self._x_to_idx(x)], dim=1).view([n_smps, 1])


class IndSmpConstantBoundedFcn(torch.nn.Module):
    """ For representing a function which assigns different bounded constant scalar values to given samples.

    This is useful, for example, when wanting to have a function which provides a different standard deviation for
    each sample in a conditional Gaussian distribution.
    """

    def __init__(self, n: int, lower_bound: float = -1.0, upper_bound: float = 1.0, init_value: float = .05):
        """
        Creates an IndSmpConstantBoundedFcn object.

        Args:

            n: The number of samples this function will assign values to.

            lower_bound, upper_bound: lower and upper bounds the function can represent.  All samples will have the same
            bounds.

            init_value: The initial value to assign to each sample.  All samples will have the same initial value.

        """
        super().__init__()

        self.n = n

        l_bounds = lower_bound*np.ones(n, dtype=np.float32)
        u_bounds = upper_bound*np.ones(n, dtype=np.float32)
        init_vls = init_value*np.ones(n, dtype=np.float32)

        self.f = ConstantBoundedFcn(lower_bound=l_bounds, upper_bound=u_bounds, init_value=init_vls)

    def forward(self, x):
        """ Assigns a value to each sample in x.

        Args:
            x: input of shape n_smps*d_x

        Returns:
            y: output of shape n_smps*1

        Raises:
            ValueError: If the number of samples in x does not match the the number of samples the function represents.
        """

        if self.n != x.shape[0]:
            raise(ValueError(' Number of input samples does not match number of output values.'))

        place_holder_input = torch.zeros(1)
        return self.f(place_holder_input).t()


class ConstantRealFcn(torch.nn.Module):
    """ Object for representing function which is constant w.r.t to input and take values anywhere in the reals.

    This is useful when working with modules which need a submodule which is a function with trainable parameters and
    you desire to use a constant in place of the function.  For example, when working with conditional distributions
    intsead of predicting the conditional mean with a neural network, you might want a constant conditional mean.
    """

    def __init__(self, init_vl: np.ndarray):
        """ Creates a ConstantRealFcn object.

        Args:
            init_vl: The initial value to initialize the function with.  The length of init_vl determines the number
            of dimensions of the output of the function.
        """

        super().__init__()

        self.n_dims = len(init_vl)

        self.vl = torch.nn.Parameter(torch.zeros(self.n_dims), requires_grad=True)
        self.vl.data = torch.from_numpy(init_vl)
        self.vl.data = self.vl.data.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Input data of shape nSmps*d_in.

        Returns:
            y: output of shape nSmps*d_out
        """

        n_smps = x.shape[0]
        return self.vl.unsqueeze(0).expand(n_smps, self.n_dims)


class IndSmpConstantRealFcn(torch.nn.Module):
    """ For representing a function which assigns different real-valued constant scalar values to given samples.

    This is useful, for example, when wanting to have a function which provides a different mean for each sample in a
    conditional Gaussian distribution.
    """

    def __init__(self, n: int, init_value: float = .01):
        """ Creates a IndSmpConstantBoundedFcn object.

        Args:
            n: The number of samples this function will assign values to.

            init_value: The initial value to assign to each sample.
        """

        super().__init__()
        self.n = n
        self.f = ConstantRealFcn(.01*np.ones(n))

    def forward(self, x):
        """ Assigns a value to each sample in x.

        Args:
            x: Input of shape n_smps*d_x

        Returns:
            y: Output of shape n_smps*1

        Raises:
            ValueError: If the number of samples in x does not match the the number of samples the function represents.
        """

        if self.n != x.shape[0]:
            raise(ValueError('Number of input samples does not match number of output samples.'))

        place_holder_input = torch.zeros(1)
        return self.f(place_holder_input).t()


class Relu(torch.nn.ModuleList):
    """ Applies a rectified linear transformation to the data y = o + relu(x + s) """

    def __init__(self, d: int):
        """ Creates a Relu object.

        Args:
            d: The dimensionality of the input and output
        """

        super().__init__()

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, std=5)
        self.register_parameter('o', o)

        s = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(s, std=5)
        self.register_parameter('s', s)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return relu(x + self.s) + self.o


class LogGaussianBumpFcn(torch.nn.Module):
    """ A module representing a log Gaussian "bump" function with trainable parameters of the form:

            y = log(g*exp(-d(x,c)),

        where d(x,c) is the distance of x from the center c defined as sqrt( (x - c)'S^-2(x-c) ), where
        S is a diagonal matrix of standard deviations.

    """

    def __init__(self, d_x: int, ctr_std_lb: float = .02, ctr_std_ub: float = 100.0, ctr_std_init: float =1.0,
                 log_gain_lb: float = -3.0, log_gain_ub: float = 0.0, log_gain_init: float =-0.05,
                 ctr_range: list = [0, 1]):
        """ Creates a LogGaussianBumpFcn object.

        Args:

            d_x: The dimensionality of the domain of the function.

            ctr_stds_lb: Lower bound center standard deviations can take on

            ctr_std_ub: Upper bound center standard deviations can take on

            ctr_stds_init: Initial value for center standard deviations.  All dimensions are initialized to the same
            value.

            log_gain_lb: Lower bound the log gain value can take on

            log_gain_ub: Upper bound the log gain value can take on

            log_gain_init: Initial value for the log gain value

            ctr_range: The range of the uniform distribution when randomly initializing the center.  All dimensions are
            selected from the same Uniform distribution.

        """

        super().__init__()

        self.ctr = torch.nn.Parameter(torch.zeros(d_x), requires_grad=True)
        torch.nn.init.uniform_(self.ctr, ctr_range[0], ctr_range[1])

        # Standard deviations determining how fast bumps fall off in each direction
        self.ctr_stds = ConstantBoundedFcn(lower_bound=np.asarray([ctr_std_lb]), upper_bound=np.asarray(ctr_std_ub),
                                           init_value=np.asarray([ctr_std_init]))

        self.log_gain_vl = ConstantBoundedFcn(lower_bound=np.asarray([log_gain_lb]), upper_bound=np.asarray([log_gain_ub]),
                                              init_value=np.asarray([log_gain_init]))

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

        if len(x_ctr_scaled.shape) > 1:
            x_dist = torch.sum(x_ctr_scaled**2, dim=1)
        else:
            x_dist = x_ctr_scaled**2

        return log_gain + -1*x_dist
