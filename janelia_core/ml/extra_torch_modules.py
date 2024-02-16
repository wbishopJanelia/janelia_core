""" Contains basic torch modules, supplementing those natviely in Torch.
"""

from typing import Sequence, Union
from warnings import warn

import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.cluster import KMeans

from janelia_core.math.basic_functions import int_to_arb_base
from janelia_core.ml.torch_fcns import knn_do

# Define aliases
OptionalTensor = Union[torch.Tensor, None]

class AffineAttentionNN(torch.nn.Module):

    def __init__(self, d_out, d_in: int = 32, n_ctrs: int = 100, distance_scales: bool = False, dropout: float = None):
        super().__init__()

        self.d_x = d_in
        self.d_y = d_out

        self.n_ctrs = n_ctrs
        self.ctrs = torch.nn.Parameter(torch.zeros(n_ctrs, d_in))
        self.Wv = torch.nn.Parameter(torch.zeros(n_ctrs, d_in, d_out))
        self.Ov = torch.nn.Parameter(torch.zeros(n_ctrs, d_out))
        
        self.s = torch.nn.Parameter(torch.zeros(d_in), requires_grad=distance_scales)
    
        self.dropout = dropout

    def init_ctrs(self, x: torch.Tensor, method='random', gain: float = 1.):
        if method == 'random':
            inds = np.random.choice(x.shape[0], self.n_ctrs)
            self.ctrs.data = x[inds] * gain
        elif method == 'fully_random':
            self.ctrs.data = torch.rand(size=self.ctrs.data.shape) * gain
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_ctrs).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_) * gain
        elif method == 'kmeans_sc':
            kmeans = KMeans(n_clusters=200).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_[np.random.choice(np.arange(0, 200), self.n_ctrs)])\
                             * gain

    def init_values(self, std_v=1e-5):
        torch.nn.init.normal_(self.Wv, std=std_v)
        # torch.nn.init.normal_(self.Ov, std=std_v)

    def _soft_assignments(self, x):

        # Select center indices based on dropout (if no dropout, use all centers)
        if self.dropout is not None and self.dropout < 1:
            ctr_inds = np.random.choice(np.arange(self.n_ctrs), int(self.dropout * self.n_ctrs), replace=False)
        else:
            ctr_inds = np.arange(self.n_ctrs)

        dist_sq = torch.sum((x.unsqueeze(1) - self.ctrs[ctr_inds]) ** 2 * self.s, dim=-1)
        assignments = torch.zeros((x.shape[0], self.n_ctrs), device=dist_sq.device)
        assignments[:, ctr_inds] = torch.nn.functional.softmax(-1 * dist_sq, dim=1)
        return assignments

    def _hard_assignments(self, x):
        assignments = torch.zeros(size=(x.shape[0], self.ctrs.shape[0]))
        assignments[np.arange(0, x.shape[0]),
                    self.assign(x)[None, :]] = 1
        return assignments

    def forward(self, x, assignment_type='soft'):

        if assignment_type == 'soft':
            assignments = self._soft_assignments(x)
        elif assignment_type == 'hard':
            assignments = self._hard_assignments(x)

        scored_weights = torch.einsum('nc,cgp -> ngp', assignments, self.Wv)
        scored_offsets = torch.matmul(assignments, self.Ov)

        return torch.sum(scored_weights * x.unsqueeze(2), dim=1) + scored_offsets

    def assign(self, x):
        _, assignments = torch.topk(self._soft_assignments(x), k=1, dim=1)
        return assignments.squeeze()
    

class AffineNearestNeighborAttentionNN(torch.nn.Module):

    def __init__(self, d_out, d_in: int = 32, n_ctrs: int = 100, k: int = None):
        super().__init__()

        self.d_x = d_in
        self.d_y = d_out
        
        if k is None:
            self.k = n_ctrs
        else:
            self.k = k

        self.n_ctrs = n_ctrs
        self.ctrs = torch.nn.Parameter(torch.zeros(n_ctrs, d_in))
        self.Wv = torch.nn.Parameter(torch.zeros(n_ctrs, d_in, d_out))
        self.Ov = torch.nn.Parameter(torch.zeros(n_ctrs, d_out))
        
    def init_ctrs(self, x: torch.Tensor, method='random', gain: float = 1.):
        if method == 'random':
            inds = np.random.choice(x.shape[0], self.n_ctrs)
            self.ctrs.data = x[inds] * gain
        elif method == 'fully_random':
            self.ctrs.data = torch.rand(size=self.ctrs.data.shape) * gain
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_ctrs).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_) * gain
        elif method == 'kmeans_sc':
            kmeans = KMeans(n_clusters=200).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_[np.random.choice(np.arange(0, 200), self.n_ctrs)])\
                             * gain

    def init_values(self, std_v=1e-5):
        torch.nn.init.normal_(self.Wv, std=std_v)
        # torch.nn.init.normal_(self.Ov, std=std_v)

    def _soft_assignments(self, x):
        dist_sq = torch.sum((x.unsqueeze(1) - self.ctrs) ** 2, dim=-1)
        closest_fcn_inds = torch.argsort(dist_sq, dim=1)[:, :self.k]
        scores = torch.nn.functional.softmax(-1*torch.gather(input=dist_sq, index=closest_fcn_inds, dim=1), dim=1)
        
        assignments = torch.zeros_like(dist_sq)
        [assignments.index_put_((torch.arange(0, x.shape[0]), closest_fcn_inds[:, i]), scores[:, i]) for i in range(self.k)]
        return assignments

    def _hard_assignments(self, x):
        assignments = torch.zeros(size=(x.shape[0], self.ctrs.shape[0]))
        assignments[np.arange(0, x.shape[0]),
                    self.assign(x)[None, :]] = 1
        return assignments

    def forward(self, x, assignment_type='soft'):

        if assignment_type == 'soft':
            assignments = self._soft_assignments(x)
        elif assignment_type == 'hard':
            assignments = self._hard_assignments(x)

        scored_weights = torch.einsum('nc,cgp -> ngp', assignments, self.Wv)
        scored_offsets = torch.matmul(assignments, self.Ov)

        return torch.sum(scored_weights * x.unsqueeze(2), dim=1) + scored_offsets

    def assign(self, x):
        _, assignments = torch.topk(self._soft_assignments(x), k=1, dim=1)
        return assignments.squeeze()


class AttentionNN(torch.nn.Module):

    def __init__(self, d_out, d_in: int = 32, n_ctrs: int = 100, distance_scales: bool = False, dropout: float = None):
        super().__init__()

        self.n_ctrs = n_ctrs
        self.ctrs = torch.nn.Parameter(torch.zeros(n_ctrs, d_in))
        self.values = torch.nn.Parameter(torch.zeros(n_ctrs, d_out))
        
        self.s = torch.nn.Parameter(torch.ones(d_in), requires_grad=distance_scales)
        
        self.dropout = dropout

    def init_ctrs(self, x: torch.Tensor, method='random', gain: float = 1.):
        if method == 'random':
            inds = np.random.choice(x.shape[0], self.n_ctrs)
            self.ctrs.data = x[inds] * gain
        elif method == 'fully_random':
            self.ctrs.data = torch.rand(size=self.ctrs.data.shape) * gain
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_ctrs).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_) * gain
        elif method == 'kmeans_sc':
            kmeans = KMeans(n_clusters=200).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_[np.random.choice(np.arange(0, 200), self.n_ctrs)])\
                             * gain

    def init_values(self, std_v=1e-5):
        torch.nn.init.normal_(self.values, std=std_v)

    def _soft_assignments(self, x):

        # Select center indices based on dropout (if no dropout, use all centers)
        if self.dropout is not None and self.dropout < 1:
            ctr_inds = np.random.choice(np.arange(self.n_ctrs), int(self.dropout * self.n_ctrs), replace=False)
        else:
            ctr_inds = np.arange(self.n_ctrs)

        dist_sq = torch.sum((x.unsqueeze(1) - self.ctrs[ctr_inds]) ** 2 * self.s, dim=-1)
        assignments = torch.zeros((x.shape[0], self.n_ctrs), device=dist_sq.device)
        assignments[:, ctr_inds] = torch.nn.functional.softmax(-1 * dist_sq, dim=1)
        return assignments

    def forward(self, x):
        assignments = self._soft_assignments(x)

        # In general, values do not have to be centers
        return torch.matmul(assignments, self.values)

    def assign(self, x):
        _, assignments = torch.topk(self._soft_assignments(x), k=1, dim=1)
        return assignments.squeeze()


class BasicExp(torch.nn.Module):
    """ Applies the transformation y = exp(x) to the data.  """

    def __init__(self):
        """ Creates a new BasicExp object. """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input. """
        return torch.exp(x)


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


class BiasAndPositiveScale(torch.nn.ModuleList):
    """ Applies a bias and non-nonegative scale transformation to the data y = abs(w)*x + o.

    Here w is the same length of x so abs(w)*x indicates element-wise product and likewise ... + o is element-wise addition.
    """

    def __init__(self, d: int, o_init_mn: float = 0.0, w_init_mn: float = 0.0,
                 o_init_std: float = .1, w_init_std: float = .1):
        """ Creates a Bias object.

        Args:
            d: The dimensionality of the input and output

            o_init_mn: The mean of the normal distribution initial biases are pulled from.

            w_init_mn: The mean of the normal distribution initial weights are pulled from.

            o_init_std: The standard deviation of the normal distribution initial biases are pulled from.

            w_init_std: The standard deviation of the normal distribution initial weights are pulled from.
        """

        super().__init__()

        w = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(w, mean=w_init_mn, std=w_init_std)
        self.register_parameter('w', w)

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, mean=o_init_mn, std=o_init_std)
        self.register_parameter('o', o)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return torch.abs(self.w)*x + self.o


class BiasAndScale(torch.nn.ModuleList):
    """ Applies a bias and scale transformation to the data y = w*x + o.

    Here w is the same length of x so w*x indicates element-wise product and likewise ... + o is element-wise addition.
    """

    def __init__(self, d: int, o_init_mn: float = 0.0, w_init_mn: float = 0.0,
                 o_init_std: float = .1, w_init_std: float = .1):
        """ Creates a Bias object.

        Args:
            d: The dimensionality of the input and output

            o_init_mn: The mean of the normal distribution initial biases are pulled from.

            w_init_mn: The mean of the normal distribution initial weights are pulled from.

            o_init_std: The standard deviation of the normal distribution initial biases are pulled from.

            w_init_std: The standard deviation of the normal distribution initial weights are pulled from.
        """

        super().__init__()

        w = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(w, mean=w_init_mn, std=w_init_std)
        self.register_parameter('w', w)

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, mean=o_init_mn, std=o_init_std)
        self.register_parameter('o', o)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return self.w*x + self.o


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

        self.set_value(init_value)

    def set_value(self, vl: np.ndarray):
        """ Sets the value of the function.

        Note: Value will be cast to a float before setting.

        Args:
            vl: The value to set the function to.

        """

        EP = 1E-7

        # Make sure everything is within bounds
        if any(vl < self.lower_bound.numpy()) or any(vl > self.upper_bound.numpy()):
            warn('Some values out of bounds.  They will be set them to bounded values.')

        y = 2*(vl - self.lower_bound.numpy())/(self.upper_bound.numpy() - self.lower_bound.numpy()) - 1
        y[y < -1] = -1
        y[y > 1] = 1

        # Make sure values we put through archtanh are not exactly -1 or 1
        y[y == -1] += EP
        y[y == 1] -= EP

        init_v = np.arctanh(y)
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

    def __init__(self, init_vl: np.ndarray, learnable_values: bool = True):
        """ Creates a ConstantRealFcn object.

        Args:
            init_vl: The initial value to initialize the function with.  The length of init_vl determines the number
            of dimensions of the output of the function.
        """

        super().__init__()

        self.n_dims = len(init_vl)

        if learnable_values:
            self.vl = torch.nn.Parameter(torch.zeros(self.n_dims), requires_grad=True)
        else:
            self.register_buffer('vl', torch.zeros(self.n_dims))
            self.dummy_param = torch.nn.Parameter(torch.empty(0))  # Dummy parameter so that we can figure out which
                                                                   # device this module is on
        self.set_vl(init_vl)

    def set_vl(self, vl: np.ndarray):
        """ Sets the value of the function.

        Note: Values will be cast to float before setting.

        Args:
            vl: The value to set the function to.
        """
        vl = torch.Tensor(vl)
        self.vl.data = vl.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Produces constant output given input.

        Args:
            x: Input data of shape nSmps*d_in.

        Returns:
            y: output of shape nSmps*d_out
        """

        n_smps = x.shape[0]
        return self.vl.unsqueeze(0).expand(n_smps, self.n_dims)


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
    

class Exp(torch.nn.ModuleList):
    """ Applies a transformation to the data y = o + exp(g*x + s) """

    def __init__(self, d: int, o_mn: float = 0.0, o_std: float = 0.1,
                               g_mn: float = 0.0, g_std: float = 0.1,
                               s_mn: float = 0.0, s_std: float = 0.1,):
        """ Creates a Exp object.

        Args:
            d: The dimensionality of the input and output

            o_mn, o_std: The mean and standard deviation for initializing o

            g_mn, g_std: The mean and standard deviation for initializing g

            s_mn, s_std: The mean and standard deviation for initializing s
        """

        super().__init__()

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, mean=o_mn, std=o_std)
        self.register_parameter('o', o)

        s = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(s, mean=s_mn, std=s_std)
        self.register_parameter('s', s)

        g = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(g, mean=g_mn, std=g_std)
        self.register_parameter('g', g)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return torch.exp(self.g*x + self.s) + self.o


class FirstAndSecondOrderFcn(torch.nn.Module):
    """ A function f(x[i]) = o[i] + sum_j a[j]*x[j] + sum_{j,k} b_[j,k]*x[j]*x[k], where o, a and b are parameters.
    """

    def __init__(self, d_in: int, d_out: int,
                 o_init_mn: float = 0, o_init_std: float = .01,
                 a_init_mn: float = 0, a_init_std: float = .01,
                 b_init_mn: float = 0, b_init_std: float = .01):
        """ Creates a new FirstAndSecondOrderFcn object.

        Args:
             d_in: The input dimensionality

             d_out: The output dimensionality

             o_init_mn, o_init_std: The mean and standard deviation of the normal distribution to
             pull initial values of o from

             a_init_mn, a_init_std: The mean and standard deviation of the normal distribution to
             pull initial values of a from

             b_init_mn, b_init_std: The mean and standard deviation of the normal distribution to
             pull initial values of b from

        """
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.o = torch.nn.Parameter(torch.zeros(d_out), requires_grad=True)
        torch.nn.init.normal_(self.o, mean=o_init_mn, std=o_init_std)

        self.a = torch.nn.Parameter(torch.zeros([d_out, d_in]), requires_grad=True)
        torch.nn.init.normal_(self.a, mean=a_init_mn, std=a_init_std)

        self.b = torch.nn.Parameter(torch.zeros([d_out, d_in, d_in]), requires_grad=True)
        torch.nn.init.normal_(self.b, mean=b_init_mn, std=b_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes intput from output.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        y = (torch.sum(x*(x.matmul(self.b)), dim=2).t() +  # Second order terms
             x.matmul(self.a.t()) +  # Linear terms
             self.o) # Offset term
        return y


class FixedOffsetCELU(torch.nn.Module):
    """ Computes y = CELU(x; alpha) + alpha + o, where o and alpha are non-learnable.

    This module simply shifts the CELU function up by alpha + 0, so that minimum value
    it can ever take on is o.

    """

    def __init__(self, alpha: float, o: float):
        """ Creates a new FixedOffsetCELU module.

        """

        super().__init__()
        self.f = torch.nn.CELU(alpha=alpha)
        self.alpha = alpha
        self.o = o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:

            x: Input

        Returns:

            y: Output, the same shape as output
        """

        return self.f(x) + self.alpha + self.o


class FixedScaleOffset(torch.nn.Module):

    def __init__(self, s: float = 1., o: float = .0):
        super().__init__()

        self.register_buffer('s', torch.Tensor([s]))
        self.register_buffer('o', torch.Tensor([o]))
        
    def forward(self, x: torch.Tensor):
        return self.s * x + self.o


class FixedOffsetExp(torch.nn.Module):
    """ Computes y = exp(x) + o, where o is a fixed, non-learnable offset. """

    def __init__(self, o: float):
        """ Creates a new FixedOffsetExp object.

        Args:
            o: The offset to apply
        """
        super().__init__()
        self.register_buffer('o', torch.Tensor([o]))

    def forward(self, x: torch.Tensor):
        """ Computes input from output.

        Args:
            x: Input tensor

        Returns:
            y: Computed output
        """

        return torch.exp(x) + self.o


class FixedOffsetAbs(torch.nn.Module):
    """ Computes y = abs(x) + o, for a fixed o. """

    def __init__(self, o: float):
        """ Creates a new FixedOffsetSq module.

        Args:
            o: The fixed offsest
        """

        super().__init__()
        self.o = o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:

            x: Input

            y: Output, the same shape as input
        """

        return torch.abs(x) + self.o


class FixedOffsetTanh(torch.nn.Module):

    """ Computes y = abs(s)*(tanh(x) + 1) + m, where s is learnable and m is fixed.

    This function can learn a different scale for each dimension of data.

    The minimum of the above function is m.  This function can be used when wanting to apply a scaled Tanh
    to values while making sure function values never go below a threshold.
    """

    def __init__(self, d: int, m: float, init_s_vls: OptionalTensor = None):
        """ Creates a new FixedOffsetTanh object. """

        super().__init__()
        self.m = m

        if init_s_vls is None:
            init_s_vls = torch.ones(d)

        self.s = torch.nn.Parameter(init_s_vls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input. """
        return torch.abs(self.s)*(torch.tanh(x) + 1) + self.m


class FormMatrixByCols(torch.nn.Module):
    """ Forms a matrix output column by column, where each column is calculated from a seperate function.

    Specifically, each column is formed by applying a module unique to that column to the same input x.
    """

    def __init__(self, col_modules: Sequence[torch.nn.Module]):
        """ Creates a new FormMatrixByCols object.

        Args:
            col_modules: col_modules[i] is the module that should be applied to form column i
        """
        super().__init__()
        self.col_modules = torch.nn.ModuleList(col_modules)

    def forward(self, x: torch.Tensor):
        """ Computes input from output. """
        return torch.cat([m(x) for m in self.col_modules], dim=1)


class IndSmpConstantBoundedFcn(torch.nn.Module):
    """ For representing a function which assigns different bounded constant scalar values to given samples.

    This is useful, for example, when wanting to have a function which provides a different standard deviation for
    each sample in a conditional Gaussian distribution.
    """

    def __init__(self, n: int, lower_bound: float = -1.0, upper_bound: float = 1.0, init_value: float = .05,
                 check_sizes: bool = True):
        """
        Creates an IndSmpConstantBoundedFcn object.

        Args:

            n: The number of samples this function will assign values to.

            lower_bound, upper_bound: lower and upper bounds the function can represent.  All samples will have the same
            bounds.

            init_value: The initial value to assign to each sample.  All samples will have the same initial value.

            check_sizes: If true, checks that the number of rows of input matches n (the number of samples) whenn
            calling forward.  If false, this check is omitted.

        """
        super().__init__()

        self.n = n

        l_bounds = lower_bound*np.ones(n, dtype=np.float32)
        u_bounds = upper_bound*np.ones(n, dtype=np.float32)
        init_vls = init_value*np.ones(n, dtype=np.float32)

        self.f = ConstantBoundedFcn(lower_bound=l_bounds, upper_bound=u_bounds, init_value=init_vls)

        self.check_sizes = check_sizes

    def forward(self, x):
        """ Assigns a value to each sample in x.

        Args:
            x: input of shape n_smps*d_x

        Returns:
            y: output of shape n_smps*1

        Raises:
            ValueError: If the number of samples in x does not match the the number of samples the function represents.
        """

        if self.check_sizes and self.n != x.shape[0]:
            raise(ValueError(' Number of input samples does not match number of output values.'))

        place_holder_input = torch.zeros(1)
        return self.f(place_holder_input).t()

    def set_value(self, vl: np.ndarray):
        """ Sets the value of the function.

        Args:

            vl: The value to set.  Must be a 1-d array of length self.n

        """

        self.f.set_value(vl)


class IndSmpConstantRealFcn(torch.nn.Module):
    """ For representing a function which assigns different real-valued constant scalar values to given samples.

    This is useful, for example, when wanting to have a function which provides a different mean for each sample in a
    conditional Gaussian distribution.
    """

    def __init__(self, n: int, init_mn: float = 0.0, init_std: float = 0.1, check_sizes: bool = True):
        """ Creates a IndSmpConstantBoundedFcn object.

        Args:
            n: The number of samples this function will assign values to.

            init_value: The initial value to assign to each sample.

            check_sizes: If true, checks that the number of rows of input matches n (the number of samples) whenn
            calling forward.  If false, this check is omitted.
        """

        super().__init__()
        self.n = n
        self.f = ConstantRealFcn(np.zeros(n))
        torch.nn.init.normal_(self.f.vl, mean=init_mn, std=init_std)

        self.check_sizes = check_sizes

    def forward(self, x):
        """ Assigns a value to each sample in x.

        Args:
            x: Input of shape n_smps*d_x

        Returns:
            y: Output of shape n_smps*1

        Raises:
            ValueError: If the number of samples in x does not match the the number of samples the function represents.
        """

        if self.check_sizes and self.n != x.shape[0]:
            raise(ValueError('Number of input samples does not match number of output samples.'))

        place_holder_input = torch.zeros(1)
        return self.f(place_holder_input).t()

    def set_value(self, vl: np.ndarray):
        """ Sets the value of the function.

        Args:
            vl: The value to set. Should be a 1-d array of length self.n
        """
        self.f.set_vl(vl)


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
    

class MultiheadAttentionNN(torch.nn.Module):

    def __init__(self, d_out, d_p: int, n_heads: int, model, model_kwargs, d_in: int = 32):
        super().__init__()

        self.Wq = torch.nn.Parameter(torch.zeros((d_in, d_p, n_heads)))
        self.Wo = torch.nn.Parameter(torch.zeros((d_p * n_heads, d_out)))

        self.networks = torch.nn.ModuleList([model(d_in=d_p, d_out=d_p, **model_kwargs) for _ in range(n_heads)])

    def init_ctrs(self, x, **kwargs):
        [nn.init_ctrs(x, **kwargs) for nn in self.networks]

    def init_values(self, **kwargs):
        [nn.init_values(**kwargs) for nn in self.networks]

    def forward(self, x: torch.Tensor):
        x_in = torch.einsum('ni,iph->nph', x, self.Wq)
        out = torch.cat([nn(x_in[..., i]) for i, nn in enumerate(self.networks)], dim=1)

        return torch.matmul(out, self.Wo)



class MultiheadStandAttentionNN(torch.nn.Module):

    def __init__(self, d_out, d_p: int, n_heads: int, d_in: int = 32, n_ctrs: int = 100):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_p = d_p
        self.n_heads = n_heads
        self.n_ctrs = n_ctrs
        
        self.ctrs = torch.nn.Parameter(torch.zeros(n_heads, n_ctrs, d_p))
        self.Wv = torch.nn.Parameter(torch.zeros(n_heads, n_ctrs, d_p, d_p))
        self.Ov = torch.nn.Parameter(torch.zeros(n_heads, n_ctrs, d_p))
        
        self.Wq = torch.nn.Parameter(torch.zeros((d_in, d_p, n_heads)))
        self.Wo = torch.nn.Parameter(torch.zeros((n_heads*d_p, d_out)))

    def init_ctrs(self, x: torch.Tensor, method='random'):
        if method == 'random':
            for i in range(self.n_heads): self.ctrs.data[i] = x[np.random.choice(x.shape[0], self.n_ctrs)][:, :self.d_p]
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_ctrs).fit(x)
            for i in range(self.n_heads): self.ctrs.data[i] = torch.Tensor(kmeans.cluster_centers_)[:, :self.d_p]
        else:
            pass

    def init_values(self, std_v=1e-5):
        torch.nn.init.normal_(self.Wv, std=std_v)
        # torch.nn.init.normal_(self.Ov, std=std_v)

    def _soft_assignments(self, x):
        dist_sq = torch.sum((x.unsqueeze(2) - self.ctrs.unsqueeze(1)) ** 2, dim=-1)
        assignments = torch.nn.functional.softmax(-1 * dist_sq, dim=2)
        return assignments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_in = torch.einsum('ni,iph->hnp', x, self.Wq)
        assignments = self._soft_assignments(x_in)

        scored_weights = torch.einsum('hnc,hcgp->hngp', assignments, self.Wv)
        scored_offsets = torch.einsum('hnc,hcp->hnp', assignments, self.Ov)

        heads_out = torch.sum(scored_weights * x_in.unsqueeze(2), dim=2) + scored_offsets
        heads_out = heads_out.reshape([x.shape[0], self.n_heads*self.d_p])

        return torch.matmul(heads_out, self.Wo)


class PWPNNFcn(torch.nn.Module):

    def __init__(self, d_out: int, d_in: int = 32, k: int = 1, order: int = 1, m=100,
                 n_fcns: int = 100, n_used_fcns: int = None):
        """ Creates a new PWLNNFcn.

        Args:

            d_in: Input dimensionality

            d_out: Output dimensionality

            k: Number of nearest neighbors to use.

            m: The number of centers to compare at once when searching for nearest neighbors.  Larger
            values use more memory but can result in significantly faster computation on GPU.

            n_used_fcns: The number of functions to use at any point in time.  Setting this equal to n_ctrs,
            results in using all centers all the time.  Setting this less than n_ctrs, will result in
            randomly dropping out some functions during each call to forward.  Setting this to None, will
            result in using all centers.

            init_ctrs: Initial centeres for each function. Of shape n_ctrs*d_in

            init_wts: Initial weights for each function. Of shape n_ctrs*d_in*d_out

            init_offsets: Initial offsets for each function. Of shape n_ctrs*d_out

        """

        super().__init__()

        self.k = k
        self.m = m
        self.n_fcns = n_fcns
        self.d_in = d_in
        self.d_out = d_out
        self.order = order

        # Set number of used functions
        if n_used_fcns is None:
            n_used_fcns = self.n_fcns
        self.n_used_fcns = n_used_fcns

        # Create parameters
        self.ctrs = torch.nn.Parameter(torch.zeros(size=(n_fcns, d_in)))
        self.wts = torch.nn.Parameter(torch.empty(size=(n_fcns, order, d_in, d_out)))
        self.offsets = torch.nn.Parameter(torch.zeros(size=(n_fcns, d_out)))

        self.init_wts()
        self.init_offsets()

    def init_ctrs(self, x: torch.Tensor, method: str = 'random'):
        if method == 'random':
            inds = np.random.choice(x.shape[0], self.n_ctrs)
            self.ctrs.data = x[inds]
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_ctrs).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_)
        else:
            raise ValueError('Initialization method undefined')

    def init_wts(self, gain: float = 1e-3):
        torch.nn.init.xavier_uniform_(self.wts, gain=gain)

    def init_offsets(self, gain: float = 1e-3):
        torch.nn.init.xavier_uniform_(self.wts, gain=gain)

    def forward(self, x: torch.Tensor):
        """ Computes output from input.

        Args:

            x: Input of shape n_smps*input_dim

        Returns:
            y: Output of shape n_smps*output_dim
        """

        # Find the k closest centers to each data point
        with torch.no_grad():
            top_k_indices = knn_do(x=x, ctrs=self.ctrs, k=self.k, m=self.m, n_ctrs_used=self.n_used_fcns)

        # Compute linear functions applied to each input data point
        selected_wts = self.wts[top_k_indices]
        selected_ctrs = self.ctrs[top_k_indices]

        applied_wts = torch.sum(selected_wts, dim=0)

        # Get power contributions according to order number for the selected centers
        order_centers = selected_ctrs[..., None] ** torch.arange(1, self.order+1, device=selected_ctrs.device)
        applied_offsets = (torch.sum(self.offsets[top_k_indices], dim=0) - torch.sum(selected_wts * order_centers.permute(0,1,3,2)[..., None], [0,2,3]))

        # Get power contributions according to order number for the input x
        order_x = x[..., None] ** torch.arange(1, self.order+1, device=x.device)
        out = torch.sum(applied_wts * order_x.permute(0,2,1)[..., None], [1,2]) + applied_offsets

        return out

    def bound(self, ctr_bounds: Sequence = [0, 1], bound_fcns: bool = True):
        """  Applies bounds to the centers.

        Bounds are applied element-wise.

        Args:

            ctr_bounds: The bounds to force centers to be between. If None, no bounds are enforced. The
            same bound is applied to all dimensions.

            bound_fcns: True if bound should be called on functiions.

        """

        if ctr_bounds is not None:
            small_inds = self.ctrs < ctr_bounds[0]
            big_inds = self.ctrs > ctr_bounds[1]
            self.ctrs.data[small_inds] = ctr_bounds[0]
            self.ctrs.data[big_inds] = ctr_bounds[1]


class PWLNNFcn(torch.nn.Module):
    """ Piecewise-linear nearest neighbor network function. """

    def __init__(self, d_out: int, d_in: int = 32, k: int = 1, m=100, n_fcns: int = 100, n_used_fcns: int = None):
        """ Creates a new PWLNNFcn.

        Args:
            
            d_in: Input dimensionality
            
            d_out: Output dimensionality

            k: Number of nearest neighbors to use.

            m: The number of centers to compare at once when searching for nearest neighbors.  Larger
            values use more memory but can result in significantly faster computation on GPU.

            n_used_fcns: The number of functions to use at any point in time.  Setting this equal to n_ctrs,
            results in using all centers all the time.  Setting this less than n_ctrs, will result in
            randomly dropping out some functions during each call to forward.  Setting this to None, will
            result in using all centers.
            
            init_ctrs: Initial centeres for each function. Of shape n_ctrs*d_in

            init_wts: Initial weights for each function. Of shape n_ctrs*d_in*d_out

            init_offsets: Initial offsets for each function. Of shape n_ctrs*d_out
            
        """

        super().__init__()

        self.k = k
        self.m = m
        self.n_fcns = n_fcns
        self.d_in = d_in
        self.d_out = d_out
        
        # Set number of used functions
        if n_used_fcns is None:
            n_used_fcns = self.n_fcns
        self.n_used_fcns = n_used_fcns

        # Create parameters
        self.ctrs = torch.nn.Parameter(torch.zeros(size=(n_fcns, d_in)))
        self.wts = torch.nn.Parameter(torch.zeros(size=(n_fcns, d_in, d_out)))
        self.offsets = torch.nn.Parameter(torch.zeros(size=(n_fcns, d_out)))
        
        self.init_wts()
        self.init_offsets()

    def init_ctrs(self, x: torch.Tensor, method: str ='random'):
        if method == 'random':
            inds = np.random.choice(x.shape[0], self.n_fcns)
            self.ctrs.data = x[inds]
        elif method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_fcns).fit(x)
            self.ctrs.data = torch.Tensor(kmeans.cluster_centers_)
        else:
            raise ValueError('Initialization method undefined')
        
    def init_wts(self, gain: float = 1e-5, offset: float = .0):
        torch.nn.init.xavier_uniform_(self.wts, gain=gain)
        self.wts.data -= torch.Tensor([offset])
        
    def init_offsets(self, gain: float = 1e-5, offset: float = .0):
        torch.nn.init.xavier_uniform_(self.offsets, gain=gain)
        self.offsets.data -= torch.Tensor([offset])

    def forward(self, x: torch.Tensor):
        """ Computes output from input.

        Args:

            x: Input of shape n_smps*input_dim

        Returns:
            y: Output of shape n_smps*output_dim
        """

        # Find the k closest centers to each data point
        with torch.no_grad():
            top_k_indices = knn_do(x=x, ctrs=self.ctrs, k=self.k, m=self.m, n_ctrs_used=self.n_used_fcns)

        # distance_weights = 1/(1 + distances**2)
        # # distance_weights /= torch.sum(distance_weights, axis=1, keepdims=True)
        # distance_weights = distance_weights[..., None, None]

        # Compute linear functions applied to each input data point
        selected_wts = self.wts[top_k_indices] #* distance_weights
        selected_ctrs = self.ctrs[top_k_indices]

        applied_wts = torch.sum(selected_wts, dim=0)
        applied_offsets = (torch.sum(self.offsets[top_k_indices], dim=0) -
                           torch.sum(torch.sum(selected_wts * selected_ctrs.unsqueeze(3), dim=0), dim=1))

        return torch.sum(applied_wts * x.unsqueeze(2), 1) + applied_offsets

    def bound(self, ctr_bounds: Sequence = [0, 1], bound_fcns: bool = True):
        """  Applies bounds to the centers.

        Bounds are applied element-wise.

        Args:

            ctr_bounds: The bounds to force centers to be between. If None, no bounds are enforced. The
            same bound is applied to all dimensions.

            bound_fcns: True if bound should be called on functiions.

        """

        if ctr_bounds is not None:
            small_inds = self.ctrs < ctr_bounds[0]
            big_inds = self.ctrs > ctr_bounds[1]
            self.ctrs.data[small_inds] = ctr_bounds[0]
            self.ctrs.data[big_inds] = ctr_bounds[1]


class QuadSurf(torch.nn.Module):
    """ A surface defined by: z = a*(x - x_0)^2 + b*(y - y_0)^2"""

    def __init__(self, ctr: torch.Tensor, coefs: torch.Tensor):
        """ Creates a new QuadSurf Module.


        Args:

            ctr: the vector [x_0, x_1]

            coefs: the vector [a, b]
        """

        super().__init__()

        self.ctr = torch.nn.Parameter(ctr)
        self.coefs = torch.nn.Parameter(coefs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*2

        Returns:
            output: Of shape n_smps*3, where each row is of the form [z, x, y], where x & y are the original x & y
            from the input
        """

        z = torch.sum(self.coefs*((x - self.ctr)**2), dim=1)
        return torch.cat([x, z.unsqueeze(1)], dim=1)


class Relu(torch.nn.ModuleList):
    """ Applies a rectified linear transformation to the data y = o + relu(x + s) """

    def __init__(self, d: int, o_mn: float = 0.0, o_std: float = .1,
                               s_mn: float = 0.0, s_std: float = .1):
        """ Creates a Relu object.

        Args:
            d: The dimensionality of the input and output

            o_mn: Mean of normal distribution for initializing offsets

            o_std: Standard deviation of normal distribution for initializing offsets

            s_mn: Mean of normal distribution for initializing shifts

            s_std: Standard deviation of normal distribution for initializing shifts
        """

        super().__init__()

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, mean=o_mn, std=o_std)
        self.register_parameter('o', o)

        s = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(s, mean=s_mn, std=s_std)
        self.register_parameter('s', s)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return relu(x + self.s) + self.o


class SCC(torch.nn.Module):
    """ A module which splits inputs, applies a function to that input (computes) and concatenates the results.

    The acronym SCC is for Split, Compute, Concatenate.  This module will:

        1) Split the input into different groups

        2) Apply a different function to each of the groups

        3) Concatenate the result

    """

    def __init__(self, group_inds: Sequence[torch.Tensor], group_modules: Sequence[torch.nn.Module]):
        """ Creates a new SCC object.

        Args:
            group_inds: group_inds[i] is tensor of dtype long indicating which input dimensions are used to
            form the data for group i.  Variables in the group will be ordered according their order in
            group_inds[i]

            group_modules: group_fcns[i] is the function to apply to data for group i.
        """

        super().__init__()

        # Register the group indices as buffers - this ensures they get moved to the appropriate device when we move
        # an SCC module
        self.group_inds = []
        for g_i, inds in enumerate(group_inds):
            buffer_name = 'grp_inds' + str(g_i)
            self.register_buffer(buffer_name, inds)
            self.group_inds.append(getattr(self, buffer_name))

        self.group_modules = torch.nn.ModuleList(group_modules)

    def forward(self, x: torch.Tensor):
        """ Computes input from output.

        Args:
            x: input of shape n_smps*d_x

            y: output of shane n_smps*d_y, where d_y is the sum of the output dimensionalities of all
            group functions.  Outputs from each group are concatenated (according to the order of the
            groups) to form y.
        """

        grp_y = [m(x[:, inds]) for inds, m in zip(self.group_inds, self.group_modules)]
        return torch.cat(grp_y, dim=1)


class SumAlongDim(torch.nn.Module):
    """ Performs a sum along a given dimension of input. """

    def __init__(self, dim: int):
        """ Create a SumAlongDim object.

        Args:
            dim: The dimension to sum along
        """

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=self.dim, keepdim=True)


class MultiDSumOfTiledHyperCubeBasisFcns(torch.nn.Module):
    """A module consisting of a collection of separate SumOfTiledHyperCubeBasisFcns that together output
     a multi-dimensional vector of outputs, where each entry of the output is produced by one function"""
    def __init__(self, d_out: int, n_divisions_per_dim: Sequence[int], dim_ranges: np.ndarray,
                 n_div_per_hc_side_per_dim: Sequence[int], init_val: float = 0., **kwargs):
        """
        Creates a MultiDSumOfTiledHyperCubeBasisFcns object.

        Args:

            n_cols: The number of columns (corresponding to the output dimensionality) in the matrices we represent
             distributions over.

            n_divisions_per_dim: n_divisions_per_dim[i] gives the number of divisions for dimension i.

            dim_ranges: The range for dimension i is dim_ranges[i,0] <= x[i] < dim_ranges[i,1]

            n_div_per_hc_side_per_dim: The number of divisions per hypercube side for each dimension

            init_val: Value with which the bump functions get initialized

        """

        super().__init__()

        self.n_cols = d_out

        col_dists = [None] * self.n_cols
        for c_i in range(self.n_cols):
            # Create the SumOfTiledHyperCubeBasisFcns object for the column
            col_dists[c_i] = SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=n_divisions_per_dim,
                                                          dim_ranges=dim_ranges,
                                                          n_div_per_hc_side_per_dim=n_div_per_hc_side_per_dim,
                                                          init_val=init_val)

        self.col_dists = torch.nn.ModuleList(col_dists)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output given input.

        Args:
            x: Input of shape n_smps*d_x.  Each x point should be within the region specified when creating teach of he
            SumOfTiledHyperCubeBasisFcns objects.

        Returns:
            y: Output of shape n_smps*n_cols.
        """
        return torch.cat([d(x) for d in self.col_dists], dim=1)

    def set_val(self, v: float):
        """ Set the output value to a single value everywhere for the distribution in each column.

        Args:
            v: The value to set the output value to
        """
        for c_i in range(self.n_cols):
            self.col_dists[c_i].set_val(v)


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

    def __init__(self, n_divisions_per_dim: Sequence[int], dim_ranges: np.ndarray,
                 n_div_per_hc_side_per_dim: Sequence[int], init_val: float = 0.):
        """
        Creates a SumOfTiledHyperCubeBasisFcns object.

        Args:

            n_divisions_per_dim: n_divisions_per_dim[i] gives the number of divisions for dimension i.

            dim_ranges: The range for dimension i is dim_ranges[i,0] <= x[i] < dim_ranges[i,1]

            n_div_per_hc_side_per_dim: The number of divisions per hypercube side for each dimension

            init_val: Value with which the bump functions get initialized

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
        n_div_per_hc_side_per_dim  = np.asarray(n_div_per_hc_side_per_dim, dtype=np.long)
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
        dim_factors = np.ones(n_dims, dtype=np.long)
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
                bump_ind_offsets[cur_chunk_start_ind:cur_chunk_end_ind] += (dim_factors[d_i]*mod_i).item()

        self.register_buffer('bump_ind_offsets', bump_ind_offsets)

        # Initialize the magnitudes of each bump function.  We initialize to zero so that if there is never
        # any training data that falls within the support of a bump, that bump will have a zero magnitude.
        # Also, we put all magnitudes in a single 1-d vector for fast indexing

        n_bump_fcns = np.cumprod(n_bump_fcns_per_dim)[-1]

        self.b_m = torch.nn.Parameter(torch.ones(n_bump_fcns)*(init_val / n_active_bump_fcns), requires_grad=True)

        self.n_active_bump_fcns = n_active_bump_fcns

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
    
    def set_val(self, v: float):
        """ Set the output value to a single value everywhere.

        Args:

            v: The value to set the output value to
        """
        cube_vl = v / self.n_active_bump_fcns
        self.b_m.data = cube_vl * torch.ones_like(self.b_m.data)


class SumOfRelus(torch.nn.Module):
    """
    A sum of Relu functions.

    The idea behind this function is we tile an input space by a collection of scaled ReLU functions, where the
    centers for each function determine the location of these functions and the weights and scales determine how they
    are oriented and the slope in the non-zero part of the relu. We then sum the results of passing a data
    point through all these functions tiling the landscape to get a final output.


    Specifically, this ia function from x \in R^d_in to y \in R^d_out, where the i^th dimensoun of output is

        y[i] = \sum_i s_ij*relu(w_ij'*(x - c_j)),

    where w_ij is a weight vector for output dimension i and relu function j, c_j is the center for relu function
    c_j and s_ij is the scale for output dimension i of relu function j.

    """

    def __init__(self, init_ctrs: torch.Tensor, init_w: torch.Tensor, init_s: torch.Tensor):
        """ Creates a new PWLManifold object.

        init_ctrs: initial centers of shape n_ctrs*input_dim

        init_w: initial weights of shape n_ctrs*output_dim*input_dim

        init_s: initial scales of shape n_ctrs*output_dim
        """

        super().__init__()

        self.ctrs = torch.nn.Parameter(init_ctrs)
        self.w = torch.nn.Parameter(init_w)
        self.s = torch.nn.Parameter(init_s)
        self.n_fcns, self.d_in = init_ctrs.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*input_dim

        Returns:
            y: Output of shape n_smps*output_dim
        """

        x_centered = x - self.ctrs.unsqueeze(1)
        relu_output = torch.relu(torch.sum(x_centered.unsqueeze(1)*self.w.unsqueeze(2), axis=-1))
        scaled_relu_output = relu_output*self.s.unsqueeze(-1)
        return torch.sum(scaled_relu_output, axis=0).t()


class SwissRole(torch.nn.Module):
    """ Represents a swiss role function.


    This is function that maps from (x,y) to (x,y,z) according to

    x = x
    y = a*(y+b)*sin(c*y)
    z = a*(y+b)*cos(c*y),

    where a, b and c are learnable parameters.

    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        """ Creates a new SwissRole object.

        Args:

            a: The a parameter.  Shuold be a 1-d vector with a single entry

            b: The b parameter.  Shuold be a 1-d vector with a single entry

            c: The c parameter.  Shuold be a 1-d vector with a single entry

        Raises:

            ValueError: If any of the parameters have the wrong shape
        """

        if a.shape != torch.Size([1]) or b.shape != torch.Size([1]) or c.shape != torch.Size([1]):
            raise(ValueError('All inputs must be 1-d tensors with a single element.'))

        super().__init__()
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.c = torch.nn.Parameter(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*2

        Returns:
            y: Output of shape n_smps*3
        """

        return torch.stack([x[:,0],
                            self.a*(x[:, 1] + self.b)*torch.sin(self.c*x[:, 1]),
                            self.a*(x[:, 1] + self.b)*torch.cos(self.c*x[:, 1])]).t()


class Tanh(torch.nn.Module):
    """ A module implementing y = s*tanh(x) + o """

    def __init__(self, d: int, o_mn: float = 0.0, o_std: float = 0.1,
                               s_mn: float = 1.0, s_std: float = 0.1,):
        """ Creates a Tanh module.

        Args:
            d: The dimensionality of the input and output

            o_mn, o_std: The mean and standard deviation for initializing o

            s_mn, s_std: The mean and standard deviation for initializing s
        """

        super().__init__()

        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, mean=o_mn, std=o_std)
        self.register_parameter('o', o)

        s = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(s, mean=s_mn, std=s_std)
        self.register_parameter('s', s)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return self.s*torch.tanh(x) + self.o


class Unsqueeze(torch.nn.Module):
    """ Wraps the torch.unsqueeze function in a module.

    Having unsqueeze in a module can be useful for when working with torch.nn.Sequential.
    """

    def __init__(self, dim:int):
        """ Creates a new Unsqueeze module.

        Args:
            dim: The index to insert the empty dimension at.
        """

        super().__init__()
        self.dim = dim

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Computes input from output. """
        return torch.unsqueeze(input=x, dim=self.dim)


class DomainBounder(torch.nn.Module):

    def __init__(self, domains):

        super().__init__()
        self.register_buffer('domains', torch.Tensor(domains))

    def forward(self, input):

        # Normalize the input
        min_values = input.min(dim=0, keepdim=True)[0]
        max_values = input.max(dim=0, keepdim=True)[0]
        normalized_input = (input - min_values) / (max_values - min_values)

        # Scale and translate to fit domain
        normalized_input *= torch.diff(self.domains, axis=1).T
        normalized_input += torch.unsqueeze(self.domains[:, 0], dim=0)

        return normalized_input

class DenseHypercube(torch.nn.Module):

    def __init__(self, d_out: int, d_in: int = 32, d_embed: int = 2, bias: bool = False):

        super().__init__()

        self.enc_linear_1 = torch.nn.Linear(in_features=d_in, out_features=20)
        self.enc_linear_2 = torch.nn.Linear(in_features=20, out_features=5)
        self.enc_linear_3 = torch.nn.Linear(in_features=5, out_features=2)
        self.encoder = torch.nn.Sequential(self.enc_linear_1,
                                           torch.nn.Tanh(),
                                           self.enc_linear_2,
                                           torch.nn.Tanh(),
                                           self.enc_linear_3,
                                           torch.nn.Tanh()
                                           )

        # Define unit domain
        domains = np.zeros(shape=(2, 2))
        domains[:, 1] = 1

        domains_marg = np.zeros(shape=(2, 2))
        domains_marg[:, 1] = .99
        domains_marg[:, 0] = .01

        # Set up domain bounder that rescales each dimension
        self.domain_bounder = DomainBounder(domains=domains_marg)

        # Set up hypercube network to learn coefficient values over the low-dimensional domain encoded
        # by the autoencoder
        self.hypercube_network = SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=[25]*d_embed,
                                                              dim_ranges=domains,
                                                              n_div_per_hc_side_per_dim=[1]*d_embed,
                                                              init_val=0.)

    def forward(self, x: torch.Tensor):
        x_enc = self.encoder(x)
        x_enc = self.domain_bounder(x_enc)
        c = self.hypercube_network(x_enc)
        return c

class DenseNetHypercube(torch.nn.Module):

    def __init__(self, d_out: int, d_in: int = 32, d_embed: int = 2, bias: bool = False):

        super().__init__()

        n_layers = 5
        growth_rate = 5
        self.densenet= DenseLNLNet(nl_class=torch.nn.Tanh, d_in=d_in, n_layers=n_layers, growth_rate=growth_rate, bias=False)
        d_dense_in = d_in + n_layers * growth_rate
        self.densenet_out = torch.nn.Linear(in_features=d_dense_in, out_features=2)
        self.encoder = torch.nn.Sequential(self.densenet, self.densenet_out)

        # Define unit domain
        domains = np.zeros(shape=(2, 2))
        domains[:, 1] = 1

        domains_marg = np.zeros(shape=(2, 2))
        domains_marg[:, 1] = .99
        domains_marg[:, 0] = .01

        # Set up domain bounder that rescales each dimension
        self.domain_bounder = DomainBounder(domains=domains_marg)

        # Set up hypercube network to learn coefficient values over the low-dimensional domain encoded
        # by the autoencoder
        self.hypercube_network = SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=[25]*d_embed,
                                                              dim_ranges=domains,
                                                              n_div_per_hc_side_per_dim=[1]*d_embed,
                                                              init_val=0.)

    def forward(self, x: torch.Tensor):
        x_enc = self.encoder(x)
        x_enc = self.domain_bounder(x_enc)
        c = self.hypercube_network(x_enc)
        return c

class DensePWLNN(torch.nn.Module):

    def __init__(self, d_out: int, d_in: int = 32, d_embed: int = 2, bias: bool = False):

        super().__init__()

        self.enc_linear_1 = torch.nn.Linear(in_features=d_in, out_features=20)
        # self.enc_linear_2 = torch.nn.Linear(in_features=20, out_features=5)
        # self.enc_linear_3 = torch.nn.Linear(in_features=5, out_features=2)
        self.encoder = torch.nn.Sequential(self.enc_linear_1,
                                           torch.nn.Tanh(),
                                           # self.enc_linear_2,
                                           # torch.nn.Tanh(),
                                           # self.enc_linear_3,
                                           # torch.nn.Tanh()
                                           )

        self.pwlnn = PWLNNFcn(d_out=2, d_in=20, k=5, m=250, n_fcns=250, n_used_fcns=None)

    def forward(self, x: torch.Tensor):
        x_enc = self.encoder(x)
        c = self.pwlnn(x_enc)
        return c


class DenseNetPWLNN(torch.nn.Module):

    def __init__(self, d_out: int, d_in: int = 32, d_embed: int = 2, bias: bool = False):

        super().__init__()

        n_layers = 5
        growth_rate = 5
        self.densenet= DenseLNLNet(nl_class=torch.nn.Tanh, d_in=d_in, n_layers=n_layers, growth_rate=growth_rate, bias=False)
        d_dense_in = d_in + n_layers * growth_rate
        self.densenet_out = torch.nn.Linear(in_features=d_dense_in, out_features=2)
        self.encoder = torch.nn.Sequential(self.densenet, self.densenet_out)

        self.pwlnn = PWLNNFcn(d_out=2, d_in=2, k=5, m=250, n_fcns=250, n_used_fcns=None)

    def forward(self, x: torch.Tensor):
        x_enc = self.encoder(x)
        c = self.pwlnn(x_enc)
        return c
