"""
Contains a class for latent regression models.

    William Bishop
    bishopw@hhmi.org
"""

import itertools
import re
import time
from typing import Sequence

import numpy as np
import torch

from janelia_core.ml.torch_distributions import CondVAEDistriubtion
from janelia_core.ml.utils import format_and_check_learning_rates


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

    A (possibly) non-linear function s_h is applied to form o_h = s_h(z_h) \in R^{d_out^h}. s_h can
    again have it's own parameters. s_h can be a general function mapping from R^{d_out^h} to R^{d_out^h},
    but in many cases, it may be a composite function which just applies the same function element-wise.

    The user can also specify pairs (g, h) when d_in^g = d_out^h, where there is a direct mapping from x_g to a
    vector v_h, v_h = c_{h,g} x_g, where c_{h,g} is a diagonal matrix.  This is most useful when x_g and
    y_g are the same set of variables (e.g, neurons) at times t-1 and t, and in addition to low-rank interactions,
    we want to include interactions between each variable and itself.

    Variables mn_h = o_h + v_h are then formed, and finally, y_h = mn_h + n_h, where n_h ~ N(0, psi_h)
    where psi_h is a diagonal covariance matrix.

    """

    def __init__(self, d_in: Sequence, d_out: Sequence, d_proj: Sequence, d_trans: Sequence,
                 m: torch.nn.Module, s: Sequence[torch.nn.Module], direct_pairs: Sequence[tuple] = None,
                 w_gain: float = 1, noise_range: Sequence[float] = [.1, .2], assign_p_u: bool = True):
        """ Create a LatentRegModel object.

        Args:

            d_in: d_in[g] gives the input dimensionality for group g of input variables.

            d_out: d_out[h] gives the output dimensionality for group h of output variables.

            d_proj: d_proj[g] gives the dimensionality for the projected latent variables for input group g.

            d_trans: d_trans[h] gives the dimensionality for the transformed latent variables for output group h.

            m: The mapping from [p_1, ..., p_G] to [t_h, ..., t_h].

            s: s[h] contains module to be applied to o_h (see above).

            direct_pairs: direct_pairs[p] contains a tuple of the form (g, h) giving a pair of input and output groups
            that should have direct connections.

            w_gain: Gain to apply to projection p and u matrices when initializing their weights.

            noise_range: Range of uniform distribution to pull psi values from during initialization.

            assign_p_u: True if p and u parameters should be created for the model.  The reason you might not want to
            do this is if you are creating a LatentRegModel for use with a framework that will fit priors over the
            columns of p and u matrices.  In this case, for the purposes of fitting, the p and u parameters of the
            LatentRegModel object are ignored, so to save memory it may be best to never create them.

        """

        super().__init__()

        # Record basic parameters
        self.d_in = d_in
        self.d_out = d_out
        self.d_proj = d_proj
        self.d_trans = d_trans
        self.direct_pairs = direct_pairs

        n_input_groups = len(d_in)
        self.n_input_groups = n_input_groups
        n_output_groups = len(d_out)
        self.n_output_groups = n_output_groups

        if assign_p_u:
            # Initialize projection matrices down
            p = [None]*n_input_groups
            for g, dims in enumerate(zip(d_in, d_proj)):
                param_name = 'p' + str(g)
                p[g] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
                torch.nn.init.xavier_normal_(p[g], gain=w_gain)
                self.register_parameter(param_name, p[g])
            self.p = p

        # Initialize projection matrices up
            u = [None]*n_output_groups
            for h, dims in enumerate(zip(d_out, d_trans)):
                param_name = 'u' + str(h)
                u[h] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
                torch.nn.init.xavier_normal_(u[h], gain=w_gain)
                self.register_parameter(param_name, u[h])
            self.u = u
        else:
            self.p = None
            self.u = None

        # Mapping from projection to transformed latents
        self.m = m

        # Direct mappings - if there are none, we set direct_mappings to None
        if direct_pairs is not None:
            n_direct_pairs = len(direct_pairs)
            direct_mappings = [None]*n_direct_pairs
            for pair_i, pair in enumerate(direct_pairs):
                c = torch.nn.Parameter(torch.ones(d_in[pair[0]]), requires_grad=True)
                torch.nn.init.normal_(c, 0, .1)
                param_name = 'c' + str(pair[0]) + '_' + str(pair[1])
                self.register_parameter(param_name, c)
                direct_mappings[pair_i] = {'pair': pair, 'c': c}
            self.direct_mappings = direct_mappings
        else:
            self.direct_mappings = None

        # Mappings from transformed latents to o_h
        self.s = torch.nn.ModuleList(s)

        # Initialize the variances for the noise variables
        psi = [None]*n_output_groups
        for h, d in enumerate(d_out):
            param_name = 'psi' + str(h)
            psi[h] = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
            torch.nn.init.uniform_(psi[h], noise_range[0], noise_range[1])
            self.register_parameter(param_name, psi[h])
        self.psi = psi

    def forward(self, x: list) -> Sequence:
        """ Computes the predicted mean from the model given input.

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        Returns:
            y: A sequence of outputs. y[h] contains the output for group h.  y[h] will be of shape n_smps*d_out[h]
        """

        return self.cond_forward(x, self.p, self.u)

    def cond_forward(self, x: list, p: list, u: list):
        """ Computes means given x and a set of projection matrices down and up.

        When this function is called, the internal p and u parameters are ignored.

        Args:

            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

            p: A sequence of tensors.  p[g] contains p_g

            u: A sequence of tensors.  u[h] contains u_h

        Returns:
            y: A sequence of outputs. y[h] contains the means for group h.  y[h] will be of shape n_smps*d_out[h]

        Raises:
            ValueError: if x is not a list
        """

        if not isinstance(x, list):
            raise(ValueError('x must be a list'))

        proj = [torch.matmul(x_g, p_g) for x_g, p_g in zip(x, p)]
        tran = self.m(proj)
        z = [torch.matmul(t_h, u_h.t()) for t_h, u_h in zip(tran, u)]

        v = [s_h(z_h) for s_h, z_h in zip(self.s, z)]

        # Add direct mappings
        if self.direct_mappings is not None:
            for dm in self.direct_mappings:
                g = dm['pair'][0]
                h = dm['pair'][1]
                v[h] = v[h] + dm['c']*x[g]

        return v

    def generate(self, x: Sequence) -> Sequence:
        """ Generates outputs from the model given inputs.

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        Returns:
            y: A sequence of generated outputs.  y[h] contains the output tensor for group h.  y[h] will be of
            shape n_smps*d_out[h]
        """

        n_output_grps = len(self.psi)

        with torch.no_grad():
            mns = self(x)
            y = [None]*n_output_grps
            for h in range(n_output_grps):
                noise_h = torch.randn_like(mns[h])*torch.sqrt(self.psi[h])
                y[h] = mns[h] + noise_h

        return y

    def neg_ll(self, y: list, mn: list, w: torch.Tensor = None) -> torch.Tensor:

        """
        Calculates the negative log likelihood of outputs given predicted means.

        The negative log likelihood of each group can be optionally be weighted.

        Args:

            y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
            shape n_smps*d_out[h]

            mns: A sequence of predicted means.  mns[h] contains the predicted means for group h.  mns[h]
            should be of shape n_smps*d_out[h]

            w: If None, no weighting of the log-likelihood for each group of outputs is performed.  If a tensor, w[i]
            is the weight for output group i. By weighting, if nll = -log P(Y_1) + -log P(Y_2) is the log likelihood
            for two ouput groups, the weighted negative log-likelihood is nll_w = -w[0]*log P(Y_0) - w[1] * log P(Y_2).

        Returns:
            The calculated negative log-likelihood for the sample

        Raises:
            ValueErorr: If y and mn are not lists
        """
        if not isinstance(y, list):
            raise(ValueError('y must be a list'))
        if not isinstance(mn, list):
            raise(ValueError('mn must be a list'))

        if w is None:
            n_grps = len(y)
            w = torch.ones(n_grps, device=self.psi[0].device)

        neg_ll = float(0)

        n_smps = y[0].shape[0]
        log_2_pi = float(np.log(2*np.pi))

        for mn_h, y_h, psi_h, w_i in zip(mn, y, self.psi, w):
            neg_ll += w_i*.5*mn_h.nelement()*log_2_pi
            neg_ll += w_i*.5*n_smps*torch.sum(torch.log(psi_h))
            neg_ll += w_i*.5*torch.sum(((y_h - mn_h)**2)/psi_h)

        return neg_ll

    def fit(self, x: Sequence[torch.Tensor], y: Sequence[torch.Tensor],
            batch_size: int=100, send_size: int=100, max_its: int=10,
            learning_rates=.01, adam_params: dict = {}, min_var: float = 0.0, update_int: int = 1000,
            parameters: list = None, l1_p_lambda: list = None, l1_u_lambda: list = None):

        """ Fits a model to data with maximum likelihood.

        This function performs stochastic optimization with the ADAM algorithm.  The weights of the model
        should be initialized before calling this function.

        Optimization will be perfomed on whatever device the model parameters are on.

        Args:

            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

            y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
            shape n_smps*d_out[h]

            batch_size: The number of samples to train on during each iteration

            send_size: The number of samples to send to the device at a time for calculating batch gradients.  It is
            most efficient to set send_size = batch_size, but if this results in computations exceeding device memory,
            send_size can be set lower.  In this case gradients will accumulated until all samples in the batch are
            sent to the device and then a step will be taken.

            max_its: The maximum number of iterations to run

            learning_rates: If a single number, this is the learning rate to use for all iteration.  Alternatively, this
            can be a list of tuples.  Each tuple is of the form (iteration, learning_rate), which gives the learning rate
            to use from that iteration onwards, until another tuple specifies another learning rate to use at a different
            iteration on.  E.g., learning_rates = [(0, .01), (1000, .001), (10000, .0001)] would specify a learning
            rate of .01 from iteration 0 to 999, .001 from iteration 1000 to 9999 and .0001 from iteration 10000 onwards.

            adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object.
            Note that if learning rate is specified here *it will be ignored.* (Use the learning_rates option instead).

            min_var: The minumum value any entry of a psi[h] can take on.  After a gradient update, values less than this
            will be clamped to this value.

            update_int: The interval of iterations we update the user on.

            parameters: If provided, only these parameters of the model will be optimized.  If none, all parameters are
            optimized.

            l1_p_lambda: The entries in the p parameters are penalized by their l1-norm the form
                l1_p_lambda[g]*sum(abs(p[g])).  If l1_p_lambda is None, no penalties will be applied.

            l1_u_lambda: Analagous to l1_p_lambda but for the u parameters.

            Raises:
                ValueError: If send_size is greater than batch_size.

            Returns:
                log: A dictionary logging progress.  Will have the enries:
                'elapsed_time': log['elapsed_time'][i] contains the elapsed time from the beginning of optimization to
                the end of iteration i

                'obj': log['obj'][i] contains the objective value at the beginning (before parameters are updated) of iteration i.

    """
        if send_size > batch_size:
            raise (ValueError('send_size must be less than or equal to batch_size.'))

        device = self.p[0].device

        if parameters is None:
            parameters = self.parameters()
        # Convert generator to list (since we need to reference parameters multiple times in the code below)
        parameters = [p for p in parameters]

        if not isinstance(learning_rates, (int, float, list)):
            raise (ValueError('learning_rates must be of type int, float or list.'))

        # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
        # where the learning rate starting at iteration 0 is guaranteed to be listed first
        learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

        optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0], **adam_params)

        n_smps = x[0].shape[0]
        cur_it = 0
        start_time = time.time()

        elapsed_time_log = np.zeros(max_its)
        obj_log = np.zeros(max_its)
        prev_learning_rate = learning_rate_values[0]

        while cur_it < max_its:
            elapsed_time = time.time() - start_time  # Record elapsed time here because we measure it from the start of
            # each iteration.  This is because we also record the nll value for each iteration before parameters are
            # updated.  In this way, the elapsed time is the elapsed time to get to a set of parameters for which we
            # report the nll.

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= cur_it)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind]
            if cur_learning_rate != prev_learning_rate:
                # We reset the whole optimizer because ADAM is an adaptive optimizer
                optimizer = torch.optim.Adam(parameters, lr=cur_learning_rate, **adam_params)
                prev_learning_rate = cur_learning_rate

            # Chose the samples for this iteration:
            cur_smps = np.random.choice(n_smps, batch_size, replace=False)
            batch_x = [x_g[cur_smps, :] for x_g in x]
            batch_y = [y_h[cur_smps, :] for y_h in y]

            # Perform optimization for this step
            optimizer.zero_grad()

            # Handle sending data to device in small chunks if needed
            start_ind = 0
            end_ind = np.min([batch_size, send_size])
            while True:
                sent_x = [batch_x_g[start_ind:end_ind, :].to(device) for batch_x_g in batch_x]
                sent_y = [batch_y_h[start_ind:end_ind, :].to(device) for batch_y_h in batch_y]

                mns = self(sent_x)
                # Calculate nll - we divide by batch size to get average (over samples) negative log-likelihood
                obj = (1 / batch_size) * self.neg_ll(sent_y, mns)
                obj.backward()

                if end_ind == batch_size:
                    break

                start_ind = end_ind
                end_ind = np.min([batch_size, start_ind + send_size])

            # Add penalties
            if l1_p_lambda is not None:
                for g, lm_g in enumerate(l1_p_lambda):
                    if lm_g != 0:
                        penalty_g = lm_g*torch.mean(torch.abs(self.p[g]))
                        penalty_g.backward()
                        obj += penalty_g

            if l1_u_lambda is not None:
                for h, lm_h in enumerate(l1_u_lambda):
                    if lm_h != 0:
                        penalty_h = lm_h*torch.mean(torch.abs(self.u[h]))
                        penalty_h.backward()
                        obj += penalty_h

            # Take a step
            optimizer.step()

            # Correct any noise variances that are too small
            with torch.no_grad():
                for psi_h in self.psi:
                    small_psi_inds = torch.nonzero(psi_h < min_var)
                    psi_h.data[small_psi_inds] = min_var

            # Log our progress
            elapsed_time_log[cur_it] = elapsed_time
            obj_vl = obj.cpu().detach().numpy()
            obj_log[cur_it] = obj_vl

            # Provide user with some feedback
            if cur_it % update_int == 0:
                print(str(cur_it) + ': Elapsed fitting time ' + str(elapsed_time) +
                      ', vl: ' + str(obj_vl) + ', lr: ' + str(cur_learning_rate))

            cur_it += 1

        # Give final fitting results (if we have not already)
        if update_int != 1:
            print(str(cur_it - 1) + ': Elapsed fitting time ' + str(elapsed_time) +
                    ', vl: ' + str(obj_vl))

        log = {'elapsed_time': elapsed_time_log, 'obj': obj_log}

        return log

    def vae_parameters(self) -> list:
        """ Returns all parameters of the model except for p and u.

        The purpose of this fuction is to return all parameters that would normally be fit when we include prior
        distributions over p and u (so p and u would not be fit directly).

        Returns:
            l: The list of paramters to fit.
        """

        all_named_params = list(self.named_parameters())
        match_inds = [re.fullmatch('^[p,u][0-9]+', p[0]) is not None for p in all_named_params]
        return [all_named_params[i] for i in range(len(all_named_params)) if match_inds[i] is False]


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


class GroupScalarTransform(torch.nn.Module):
    """ Mapping which forms output by multiplying each entry in group input vectors by a seperate scalar."""

    def __init__(self, d: Sequence[int]):
        """ Creates a GroupScalarTransform object.

        Args:
           d: d[i] gives the dimensionality of group i
        """
        super().__init__()

        n_grps = len(d)

        self.v = [None]*n_grps
        for g in range(n_grps):
            param_name = 'v' + str(g)
            self.v[g] = torch.nn.Parameter(torch.ones(d[g]), requires_grad=True)
            self.register_parameter(param_name, self.v[g])

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """" Computes output given input.

        Args:
            x: Input. x[g] gives the input for group g as a tensor of shape n_smps*n_dims

        Returns:
            y: Output. y[g] gives the output for group g as a tensor of shampe n_smps*n_dims

        """

        return [x_g*v_g for v_g, x_g in zip(self.v, x)]


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


def vae_fit_latent_reg_model(l_mdl: LatentRegModel, q_p_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             q_u_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             prior_p_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             prior_u_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             x: Sequence[torch.Tensor], y: Sequence[torch.Tensor], x_props: Sequence[torch.Tensor],
                             y_props: Sequence[torch.Tensor], batch_size: int=100, send_size: int=100, max_its: int=10,
                             learning_rates=.01, adam_params: dict = {}, min_var: float=0.0, update_int: int=100,
                             fit_priors: bool = True, grp_w: torch.Tensor = None):

    """ A function for fitting a latent regression model and a prior over it's modes with variational inference.

    Note: When calling this function the values of the p and u parameters of the latent regression model are
    ignored (since these represent point values and this function fits a distribution over the modes).

    Note: This function will move batches of data to whatever device the latent regression and distribution parameters
    are on.  (We implicitly assume the latent regression model and all mode distributions are on the same device.)
    Property data will also be moved to this device.

    Args:

        l_mdl: The latent regression model to fit.

        q_p_dists, q_u_dists: Inference distributions for each mode.  q_p_dists[g][j] is the distribution over the j^th
        column of the p_g matrix of a LatentRegModel. q_u_dists[h][j] is the same for the j^th column of the u_h matrix.

        prior_p_dists, prior_u_dists: Prior distributions for each mode.  prior_p_dists[g][j] is the distribution over
        the j^th column of p_g, and prior_u_dists[h][j] is the same for the j^th column of u_h.

        x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
        shape n_smps*d_out[h]

        x_props: A sequence of properties for variables in x.  (E.g., if x is neural activity, this is the
        properties, such as position, for each neuron.)  x_props[g][j,:] contains the properties for variable j in
        group g.

        y_props: A sequence of properties for variables in y. y_props[h][j,:] contains the properties for variable j
        in group h.

        batch_size: The number of samples to train on during each iteration

        send_size: The number of samples to send to the device at a time for calculating batch gradients.  It is
        most efficient to set send_size = batch_size, but if this results in computations exceeding device memory,
        send_size can be set lower.  In this case gradients will accumulated until all samples in the batch are
        sent to the device and then a step will be taken.

        max_its: The maximum number of iterations to run

        learning_rates: If a single number, this is the learning rate to use for all iteration.  Alternatively, this
        can be a list of tuples.  Each tuple is of the form (iteration, learning_rate), which gives the learning rate
        to use from that iteration onwards, until another tuple specifies another learning rate to use at a different
        iteration on.  E.g., learning_rates = [(0, .01), (1000, .001), (10000, .0001)] would specify a learning
        rate of .01 from iteration 0 to 999, .001 from iteration 1000 to 9999 and .0001 from iteration 10000 onwards.

        adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object.
        Note that if learning rate is specified here *it will be ignored.* (Use the learning_rates option instead).

        min_var: The minumum value any entry of a psi[h] can take on.  After a gradient update, values less than this
        will be clamped to this value.

        update_int: The interval that updates should be printed

        fit_priors: If false, fitting is done where the only term that is optimized is negative log-likelihood (KL
        divergence between priors and q is omitted).  Setting this to false, may be helpful for doing an initial run to
        fit model parameters.

        grp_w: grp_w[i] is the weight to apply for group i when summing log-likelihoods across groups.
        If None, no weighting will be used.

    Returns:
        log: A dictionary logging progress.  Will have the enries:
            'elapsed_time': log['elapsed_time'][i] contains the elapsed time from the beginning of optimization to
            the end of iteration i

            'obj': log['obj'][i] contains the objective value at the beginning (before parameters are updated) of
            iteration i.

    Raises:

        ValueError: If send_size is larger than batch size

    """

    if send_size > batch_size:
        raise(ValueError('send_size must be less than or equal to batch size'))

    # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
    # where the learning rate starting at iteration 0 is guaranteed to be listed first
    learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

    # Get a list of all the parameters we need to fit
    subject_params = l_mdl.vae_parameters()
    subject_params = [p[1] for p in subject_params] # Keep only parameters, discarding names
    q_p_params = itertools.chain(*[d.r_params() for q_p_g_dists in q_p_dists for d in q_p_g_dists])
    q_u_params = itertools.chain(*[d.r_params() for q_u_g_dists in q_u_dists for d in q_u_g_dists])
    prior_p_params = itertools.chain(*[d.r_params() for prior_p_g_dists in prior_p_dists for d in prior_p_g_dists])
    prior_u_params = itertools.chain(*[d.r_params() for prior_u_g_dists in prior_u_dists for d in prior_u_g_dists])
    parameters = list(itertools.chain(subject_params, q_p_params, q_u_params, prior_p_params, prior_u_params))

    # See what device parameters are on
    device = parameters[0].device
    run_on_gpu = device.type == 'cuda'

    # See what memory usage on GPU is before we've moved any data there
    if run_on_gpu:
        init_gpu_mem_usage = torch.cuda.memory_allocated()
        print('Initial GPU memory usage: ' + str(init_gpu_mem_usage) + ' bytes.')

    # Move property data to the device
    x_props = [props.to(device) for props in x_props]
    y_props = [props.to(device) for props in y_props]

    # See what memory usage on GPU is after sending neuron properties
    if run_on_gpu:
        after_props_gpu_mem_usage = torch.cuda.memory_allocated()
        print('GPU memory usage after sending properties: ' + str(after_props_gpu_mem_usage) + ' bytes.')

    # Setup optimizer
    optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0], **adam_params)

    # Calculate the correction factor we apply when calculating negative
    # log-likelihood to account for the fact that our batch sizes don't use
    # all samples - this is to prevent batch_size as effectively acting
    # as a tuning parameter (because if we don't apply this correction, smaller
    # batch sizes will mean the priors have more influence - we want the influence
    # of the priors to be determined only by the total amount of data we fit on
    n_smps = x[0].shape[0]
    batch_ratio = float(n_smps)/batch_size

    # See how many input and output groups we have
    n_input_grps = len(q_p_dists)
    n_output_grps = len(q_u_dists)

    # Setup variables we will need for fitting
    cur_it = 0
    start_time = time.time()
    prev_learning_rate = learning_rate_values[0]

    elapsed_time_log = np.zeros(max_its)
    elbo_log = np.zeros(max_its)

    # Perform fitting
    while cur_it < max_its:

            elapsed_time = time.time() - start_time  # Record elapsed time here because we measure it from the start of
            # each iteration.  This is because we also record the objective value for each iteration before parameters
            # are updated.  In this way, the elapsed time is the elapsed time to get to a set of parameters for which we
            # report the objective value

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= cur_it)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind]
            if cur_learning_rate != prev_learning_rate:
                # We reset the whole optimizer because ADAM is an adaptive optimizer
                optimizer = torch.optim.Adam(parameters, lr=cur_learning_rate, **adam_params)
                prev_learning_rate = cur_learning_rate

            # Chose the data samples for this iteration:
            cur_smps = np.random.choice(n_smps, batch_size, replace=False)
            batch_x = [x_g[cur_smps, :] for x_g in x]
            batch_y = [y_h[cur_smps, :] for y_h in y]

            # Zero the gradients to prepare for this optimization step
            optimizer.zero_grad()

            # Sample from the q distribution
            q_p_smps = [[d.form_standard_sample(d.sample(g_props))for d in q_p_g_dists]
                        for q_p_g_dists, g_props in zip(q_p_dists, x_props)]
            q_p_smps_t = [torch.cat(smps_g, dim=1) for smps_g in q_p_smps]

            q_u_smps = [[d.form_standard_sample(d.sample(g_props)) for d in q_u_g_dists]
                        for q_u_g_dists, g_props in zip(q_u_dists, y_props)]
            q_u_smps_t = [torch.cat(smps_g, dim=1) for smps_g in q_u_smps]

            # Send data to device in small chunks (if send_size < batch_size) to calculate negative log-likelhood
            start_ind = 0
            end_ind = np.min([batch_size, send_size])
            neg_ll = 0
            elbo_db = 0
            while True:
                sent_x = [batch_x_g[start_ind:end_ind, :].to(device) for batch_x_g in batch_x]
                sent_y = [batch_y_h[start_ind:end_ind, :].to(device) for batch_y_h in batch_y]

                sent_y_hat = l_mdl.cond_forward(x=sent_x, p=q_p_smps_t, u=q_u_smps_t)
                sent_nll = batch_ratio*l_mdl.neg_ll(y=sent_y, mn=sent_y_hat, w=grp_w)

                elbo_db += sent_nll
                #sent_nll.backward(retain_graph=True)

                # We call backward on each sent chunk of data but we still need to accumulate our
                # total negative log likelihood term for the elbo
                neg_ll += sent_nll.detach().cpu().numpy()

                if end_ind == batch_size:
                    break

                start_ind = end_ind
                end_ind = np.min([batch_size, start_ind + send_size])

            if fit_priors:
                # Calculate kl divergence between conditional posterior and priors for p modes
                kl_p = [None]*n_input_grps # Keep track of KL divergence for each mode for diagnostic purposes
                for g in range(n_input_grps):
                    q_p_mode_dists = q_p_dists[g]
                    prior_p_mode_dists = prior_p_dists[g]

                    n_p_mode_dists = len(q_p_mode_dists)
                    p_mode_kls = np.zeros(n_p_mode_dists)
                    for m_i in range(n_p_mode_dists):
                        mode_kl = torch.sum(q_p_mode_dists[m_i].kl(d_2=prior_p_mode_dists[m_i], x=x_props[g],
                                                               smp=q_p_smps[g][m_i]))
                        elbo_db += mode_kl
                        #mode_kl.backward()
                        p_mode_kls[m_i] = mode_kl.detach().cpu().numpy()
                    kl_p[g] = p_mode_kls

                # Calculate kl divergence between conditional posterior and priors for u modes
                kl_u = [None]*n_output_grps
                for h in range(n_output_grps):
                    q_u_mode_dists = q_u_dists[h]
                    prior_u_mode_dists = prior_u_dists[h]

                    n_u_mode_dists = len(q_u_mode_dists)
                    u_mode_kls = np.zeros(n_u_mode_dists)
                    for m_i in range(n_u_mode_dists):
                        mode_kl = torch.sum(q_u_mode_dists[m_i].kl(d_2=prior_u_mode_dists[m_i], x=y_props[h],
                                                                smp=q_u_smps[h][m_i]))
                        elbo_db += mode_kl
                        #mode_kl.backward()
                        u_mode_kls[m_i] = mode_kl.detach().cpu().numpy()
                    kl_u[h] = u_mode_kls

            # Take a step here
            elbo_db.backward()
            optimizer.step()

            # Calculate the value of the ELBO here
            if fit_priors:
                kl_p_sum = np.sum([np.sum(kl_p_g) for kl_p_g in kl_p])
                kl_u_sum = np.sum([np.sum(kl_u_g) for kl_u_g in kl_u])
            else:
                kl_p_sum = 0
                kl_u_sum = 0
            neg_elbo = neg_ll + kl_p_sum + kl_u_sum

            # Correct any noise variances that are too small
            with torch.no_grad():
                for psi_h in l_mdl.psi:
                    small_psi_inds = torch.nonzero(psi_h < min_var)
                    psi_h.data[small_psi_inds] = min_var

            # Log our progress
            elapsed_time_log[cur_it] = elapsed_time
            elbo_log[cur_it] = -1*neg_elbo

            # Provide user with some feedback of requested
            if cur_it % update_int == 0:
                if run_on_gpu:
                    cur_gpu_mem_usage = torch.cuda.memory_allocated()
                else:
                    cur_gpu_mem_usage = np.nan
                print('It: ' + str(cur_it) + ': Elapsed fitting time ' + str(elapsed_time) +
                      ', elbo: ' + str(-1*neg_elbo) + ', lr: ' + str(cur_learning_rate) +
                      ', GPU mem. usage: ' + str(cur_gpu_mem_usage) + ' bytes')
                print('    ll: ' + str(-1*neg_ll) + ', kl_p_sum: ' + str(kl_p_sum) + ', kl_u_sum: ' + str(kl_u_sum))

            cur_it += 1

    # Give final fitting results (if we have not already)
    if (cur_it - 1) % update_int != 0:
        print('It: ' + str(cur_it - 1) + ': Elapsed fitting time ' + str(elapsed_time) +
              ', elbo: ' + str(-1*neg_elbo) + ', lr: ' + str(cur_learning_rate) +
              ', GPU mem. usage: ' + str(cur_gpu_mem_usage) + ' bytes')
        print('    ll: ' + str(-1*neg_ll) + ', kl_p_sum: ' + str(kl_p_sum) + ', kl_u_sum: ' + str(kl_u_sum))

    # Format output
    log = {'elapsed_time': elapsed_time_log, 'elbo': elbo_log}

    return log
