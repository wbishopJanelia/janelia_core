""" Contains basic torch modules, supplementing those natviely in Torch.
"""

import numpy as np
import torch
from torch.nn.functional import relu


class Bias(torch.nn.ModuleList):
    """ Applies a bias transformation to the data y = x + o """

    def __init__(self, d: int):
        """ Creates a Bias object.

        Args:
            d: The dimensionality of the input and output
        """

        super().__init__()
        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, std=5)
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
            should be arrays providing a the bounds for each dimension of output.

            init_value: If provided, this is the constant output the function is initialized to.  Should be an
            array providing initial values for each dimension. If not provided, the constant value will be initialized
            to be halfway between the lower and upper bound.
        """

        super().__init__()

        n_dims = len(lower_bound)
        self.n_dims = n_dims

        self.lower_bound = torch.tensor(lower_bound)
        self.lower_bound = self.lower_bound.float()
        self.upper_bound = torch.tensor(upper_bound)
        self.upper_bound = self.upper_bound.float()

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
