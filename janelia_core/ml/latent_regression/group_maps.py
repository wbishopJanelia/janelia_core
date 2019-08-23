""" Objects for mapping input groups of variables to output groups of variables.

    The maps contained in this module are intended to serve as mappings in the latent spaces of latent regression
    models.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

import numpy as np
import torch


class LinearMap(torch.nn.Module):
    """ Wraps torch.nn.Linear for use with a latent mapping.

     All inputs are concatenated together before being passed through the mapping to form output.

     """

    def __init__(self, d_in: Sequence, d_out: Sequence, bias=False):
        """ Creates a LinearMap object.

        Args:
            d_in: d_in[g] gives the dimensionality of input group g

            d_out: d_out[g] gives the dimensionality of output group g

            bias: True, if the linear mapping should include a bias.

        """
        super().__init__()

        # We compute the indices to get the output variables from a concatentated vector of output
        n_output_groups = len(d_out)
        out_slices = [None]*n_output_groups
        start_ind = 0
        for h in range(n_output_groups):
            end_ind = start_ind + d_out[h]
            out_slices[h] = slice(start_ind, end_ind, 1)
            start_ind = end_ind
        self.out_slices = out_slices

        # Setup the linear network
        d_in_total = np.sum(d_in)
        d_out_total = np.sum(d_out)
        self.nn = torch.nn.Linear(d_in_total, d_out_total, bias=bias)

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Computes output given input.

        Args:
            x: Input.  x[g] gives the input for input group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output.  y[h] gives the output for output group h as a tensor or shape n_smps*n_dims
        """

        x_conc = torch.cat(x, dim=1)
        y_conc = self.nn(x_conc)
        return [y_conc[:, s] for s in self.out_slices]


class NNMap(torch.nn.Module):
    """ Wraps a general neural network for use with a latent mapping.

    All inputs are concatenated together before being passed through the mapping to form output.

    The transformed output can then be partitioned to form different output groups.
    """

    def __init__(self, d_out: Sequence[int], nn: torch.nn.Module):
        """
        Args:
            d_out: d_out[h] gives the dimensionality of output group h.  Outputs are read in the order of groups.  So
            for example, the first d_out[0] outputs will be outputs for group 0, then next d_out[1] outputs for group
            1 and so on...

            nn: The neural network to wrap.  Should accept inputs of size equal to the concatenation of all input
            and produce outputs of the size of all concatenated output.
        """

        super().__init__()

        # We compute the indices to get the output variables from a concatentated vector of output
        n_output_groups = len(d_out)
        out_slices = [None]*n_output_groups
        start_ind = 0
        for h in range(n_output_groups):
            end_ind = start_ind + d_out[h]
            out_slices[h] = slice(start_ind, end_ind, 1)
            start_ind = end_ind
        self.out_slices = out_slices

        self.nn = nn

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Computes output given input.

        Args:
            x: Input.  x[g] gives the input for input group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output.  y[h] gives the output for output group h as a tensor or shape n_smps*n_dims
        """

        x_conc = torch.cat(x, dim=1)
        y_conc = self.nn(x_conc)
        return [y_conc[:, s] for s in self.out_slices]


class IdentityMap(torch.nn.Module):
    """ Identity latent mapping."""

    def __init__(self):
        """ Creates an IdentityMap object. """
        super().__init__()

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Passes through input as output.

        Args:
            x: Input.  x[g] gives the input for input group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output.  y[g] gives the output for output group h as a tensor or shape n_smps*n_dims
        """

        return x


class ElementWiseTransformedGroupLatents(torch.nn.Module):
    """ Output is formed by applying a function elementwise to input for each group. """

    def __init__(self, f: torch.nn.Module):
        """ Creates an ElementWiseTransformedGroupLatents object.

        Args:
            f: The function to apply to each element of output for each group.
        """

        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input. """
        return [self.f(x_i) for x_i in x]


class GroupLinearTransform(torch.nn.Module):
    """ Mapping which forms output by multiplying each entry in group input vectors by a seperate scalar.

    Offsets can also optionally be added.
    """

    def __init__(self, d: Sequence[int], offsets: bool = False,
                 v_mn: float = 0.0, v_std: float = 0.1,
                 o_mn: float = 0.0, o_std: float = 0.1):
        """ Creates a GroupScalarTransform object.

        Args:
           d: d[i] gives the dimensionality of group i

           offsets: True if offsets should also be included.

           v_mn, v_std: The mean and standard deviation for initializing the slope of the linear mappings

           o_mn, o_std: The mean and standard deviations for initializing the offsets of the linear mappings
        """
        super().__init__()

        n_grps = len(d)

        self.v = [None]*n_grps
        self.o = [None]*n_grps
        for g in range(n_grps):
            param_name = 'v' + str(g)
            self.v[g] = torch.nn.Parameter(torch.zeros(d[g]), requires_grad=True)
            torch.nn.init.normal_(self.v[g], mean=v_mn, std=v_std)
            self.register_parameter(param_name, self.v[g])

            param_name = 'o' + str(g)
            self.o[g] = torch.nn.Parameter(torch.zeros(d[g]), requires_grad=True)
            torch.nn.init.normal_(self.o[g], mean=o_mn, std=o_std)
            self.register_parameter(param_name, self.o[g])

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """" Computes output given input.

        Args:
            x: Input. x[g] gives the input for group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output. y[g] gives the output for group g as a tensor of shampe n_smps*n_dims

        """

        return [x_g*v_g + o_g for v_g, o_g, x_g in zip(self.v, self.o, x)]


class GroupMatrixMultiply(torch.nn.Module):
    """ Mapping which applies a matrix multiply seperately to each input vector to form output vectors."""

    def __init__(self, d_in: Sequence[int], d_out: Sequence[int], w_gain: float = 1.0):
        """ Creates a GroupMatrixMultiply object.

        Args:
            d_in: d_in[i] gives the input dimension of group i

            d_out: d_out[i] gives the output dimension of group i

            w_gain: Gain to apply when initializing matrix weights

        """

        super().__init__()

        n_grps = len(d_in)

        self.n_grps = n_grps
        self.d_in = d_in
        self.d_out = d_out

        self.w = [None]*n_grps
        for g, dims in enumerate(zip(d_in, d_out)):

            d_i = dims[0]
            d_o = dims[1]

            w_g = torch.nn.Parameter(torch.zeros(d_i, d_o), requires_grad=True)
            torch.nn.init.xavier_normal_(w_g, gain=w_gain)
            self.w[g] = w_g

            param_name = 'w_' + str(g)
            self.register_parameter(param_name, w_g)

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Computes output given input.

        Args:
            x: Input. x[g] gives the input for group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output. y[g] gives the output for group g as a tensor of shampe n_smps*n_dims
        """

        return [torch.matmul(x_g, w_g.t()) for w_g, x_g in zip(self.w, x)]


class ConcatenateMap(torch.nn.Module):
    """ Mapping which concatenates input to form output.

    Concatenation follows the order of input groups.
    """

    def __init__(self, conc_grps: np.ndarray):
        """ Creates a ConcatenateMap object.

        Args:
            conc_grps: A binary array.  conc_grps[h, :] is a vector indicating which groups of projected
            latents should be concatenated to form output for tran_h.

        Raises:
            ValueError: If dtype of conc_grps is not bool

        """

        super().__init__()

        if not (conc_grps.dtype == np.dtype('bool')):
            raise(ValueError('dtype of conc_grps must be bool'))
        self.conc_grps = conc_grps

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Concatents input to form output. """

        n_output_grps, n_input_grps = self.conc_grps.shape

        y = [torch.cat(tuple(x[g] for g in range(n_input_grps) if self.conc_grps[h, g] == True), dim=1)
             for h in range(n_output_grps)]

        return y


class ConcatenateAndSelectMap(torch.nn.Module):
    """ Concatenates input from all input groups and then selects entries of the result for different output groups.
    """

    def __init__(self, d_in: Sequence[int], output_inds: Sequence[Sequence[torch.Tensor]]):
        """ Creates a ConcatenateAndSelectMap object.

        Args:
            d_in: d_in[g] is the dimensionality of the g^th input group

            output_inds: output_inds[h][g] is a torch tensor of dtype long indicating which entries of
            the g^th input group should be in the h^th output group.  The output of each group will be formed
            by concatenating the included entries for all input groups.  The order of concatenating groups follows
            the order of the input groups. The value None indicates no entries from input g should be selected.

        """

        super().__init__()

        # Form indices for selecting input
        n_input_grps = len(d_in)
        n_output_grps = len(output_inds)
        input_offsets = np.zeros(n_input_grps)

        for g in range(1, n_input_grps):
            input_offsets[g] = input_offsets[g-1] + d_in[g - 1]
        input_offsets = input_offsets.astype('long')

        sel_tensors = [torch.cat([inds + input_offsets[g] for g, inds in enumerate(input_grp_inds) if inds is not None])
                       for input_grp_inds in output_inds]

        # Register the indices for selecting input as buffers
        self.sel_tensors = []
        for h, sel_t in enumerate(sel_tensors):
            buffer_name = 'sel_tensor' + str(h)
            self.register_buffer(buffer_name, sel_t)
            self.sel_tensors.append(getattr(self, buffer_name))

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """ Forms output from input.

        Args:
            x: x[g] is the input for the g^th group of shape n_smps*d_g

        Returns:
            y: y[h] is the output for the h^th group of shape n_smps*d_h
        """

        # Concatenate all input
        x_conc = torch.cat(x, dim=1)
        return [x_conc[:, inds] for inds in self.sel_tensors]




