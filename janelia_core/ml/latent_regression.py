"""
Contains a class for latent regression models.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

import numpy as np
import torch


class LatentRegModel(torch.nn.Module):
    """ A latent variable regression model.

    In this model, we have G groups of input variables, x_g \in R^{d_in^g} for g = 1, ..., G and
    H groups of output variables, y_h \in R^{d_out^h} for h = 1, ..., H

    We form G groups of "projected" latent variables as proj_g = p_g^T x_g, for proj_g \in R^{d_proj^g},
    Note that p_g need not be an orthonormal projection.

    There are also H sets of "transformed" latent variables, tran_1, ..., tran_H, with tran_h \in R^{d_trans^h}, where
    d_trans^h is the dimensionality of the transformed latent variables for group h.

    Each model is equipped with a mapping, m, from [proj_1, ..., proj_G] to [tran_1, ..., tran_G].  The mapping m may
    have it's own parameters.  The function m.forward() should accept a list, [proj_1, ..., proj_G], as input where
    proj_g is a tensor of shape n_smps*d_proj^g and should output a list, [tran_1, ..., tran_G], where trah_h is a
    tensor of shape n_smps*d_trans^h.

    The transformed latents are mapped to a high-dimensional vector z_h = u_h tran_h, where z_h \in R^{d_out^h}.
    A (possibly) non-linear function s_his applied element-wise to form mn_h = s_h(z_h) \in R^{d_out^h}. s_g can
    again have it's own parameters.

    Finally, y_h = mn_h + n_h, where n_h ~ N(0, psi_h) where psi_h is a diagonal covariance matrix.


    """

    def __init__(self, d_in: Sequence, d_out: Sequence, d_proj: Sequence, d_trans: Sequence,
                 m: torch.nn.Module, s: Sequence[torch.nn.Module]):
        """ Create a LatentRegModel object.

        Args:

            d_in: d_in[g] gives the input dimensionality for group g of input variables.

            d_out: d_out[h] gives the output dimensionality for group h of output variables.

            d_proj: d_proj[g] gives the dimensionality for the projected latent variables for input group g.

            d_trans: d_trans[h] gives the dimensionality for the transformed latent variables for output group h.

            m: The mapping from [p_1, ..., p_G] to [t_1, ..., t_G].

            s: s[g] contains the function to be applied element-wise to l_g (see above).

        """

        super().__init__()

        # Initialize projection matrices down
        n_input_groups = len(d_in)
        p = [None]*n_input_groups
        for g, dims in enumerate(zip(d_in, d_proj)):
            param_name = 'p' + str(g)
            p[g] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
            self.register_parameter(param_name, p[g])
        self.p = p

        # Initialize projection matrices up
        n_output_groups = len(d_out)
        u = [None]*n_output_groups
        for h, dims in enumerate(zip(d_out, d_trans)):
            param_name = 'u' + str(h)
            u[h] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
            self.register_parameter(param_name, u[h])
        self.u = u

        # Mapping from projection to transformed latents
        self.m = m

        # Mappings from transformed latents to means
        self.s = s

        # Initialize the variances for the noise variables
        for g, d in enumerate(d_out):
            param_name = 'psi' + str(g)
            psi = torch.nn.Parameter(torch.zeros([d, 1]), requires_grad=True)
            self.register_parameter(param_name, psi)

    def forward(self, x: Sequence) -> Sequence:
        """ Computes the predicted mean from the model given input.

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        Returns:
            y: A sequence of outputs. y[h] contains the output for group h.  y[h] will be of shape n_smps*d_out[h]
        """

        proj = [torch.matmul(x_g, p_g) for x_g, p_g in zip(x, self.p)]
        tran = self.m(proj)
        z = [torch.matmul(t_h, u_h.t()) for t_h, u_h in zip(tran, self.u)]
        mu = [s_h(z_h) for z_h, s_h in zip(z, self.s)]
        return mu


class LinearMap(torch.nn.Module):
    """ Wraps torch.nn.Linear for use a latent mapping. """

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









