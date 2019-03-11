"""
Contains a class for latent regression models.

    William Bishop
    bishopw@hhmi.org
"""

import time
from typing import Sequence

import numpy as np
import torch

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

    In addition, the user can specify pairs (g, h) when d_in^g = d_out^h, where there is a direct mapping for the
    from x_g to a vector v_h, v_h = c_{h,g} x_g, where c_{h,g} is a diagonal matrix.  This is most useful when x_g and
    y_g are the same set of variables (e.g, neurons) at times t-1 and t, and in addition to low-rank interactions,
    we want to include interactions between each variable and itself.

    Variables o_h = z_h + v_h are then formed (if there is an h for which v_h is not computed, then o_h = z_h.)

    A (possibly) non-linear function s_h is applied to form mn_h = s_h(o_h) \in R^{d_out^h}. s_h can
    again have it's own parameters. s_h can general function mapping from R^{d_out^h} to R^{d_out^h},
    but in many cases, it may be a composite function which just applies the same function to o_h, element-wise.

    Finally, y_h = mn_h + n_h, where n_h ~ N(0, psi_h) where psi_h is a diagonal covariance matrix.

    """

    def __init__(self, d_in: Sequence, d_out: Sequence, d_proj: Sequence, d_trans: Sequence,
                 m: torch.nn.Module, s: Sequence[torch.nn.Module], direct_pairs: Sequence[tuple] = None,
                 w_gain: float = 1, noise_range: Sequence[float] = [.1, .2]):
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

        """

        super().__init__()

        # Initialize projection matrices down
        n_input_groups = len(d_in)
        self.n_input_groups = n_input_groups
        p = [None]*n_input_groups
        for g, dims in enumerate(zip(d_in, d_proj)):
            param_name = 'p' + str(g)
            p[g] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
            torch.nn.init.xavier_normal_(p[g], gain=w_gain)
            self.register_parameter(param_name, p[g])
        self.p = p

        # Initialize projection matrices up
        n_output_groups = len(d_out)
        self.n_output_groups = n_output_groups
        u = [None]*n_output_groups
        for h, dims in enumerate(zip(d_out, d_trans)):
            param_name = 'u' + str(h)
            u[h] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
            torch.nn.init.xavier_normal_(u[h], gain=w_gain)
            self.register_parameter(param_name, u[h])
        self.u = u

        # Mapping from projection to transformed latents
        self.m = m

        # Direct mappings - there are none, we set direct_mappings to None
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

        # Mappings from transformed latents to means
        self.s = torch.nn.ModuleList(s)

        # Initialize the variances for the noise variables
        psi = [None]*n_output_groups
        for h, d in enumerate(d_out):
            param_name = 'psi' + str(h)
            psi[h] = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
            torch.nn.init.uniform_(psi[h], noise_range[0], noise_range[1])
            self.register_parameter(param_name, psi[h])
        self.psi = psi

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

        # Add direct mappings, while variable names are in general the same as in the __init__ documentation, here
        # the variable name o is omitted in favor of adding to z to keep the code simple
        if self.direct_mappings is not None:
            for dm in self.direct_mappings:
                g = dm['pair'][0]
                h = dm['pair'][1]
                z[h] = z[h] + dm['c']*x[g]

        mn = [s_h(z_h) for z_h, s_h in zip(z, self.s)]

        return mn

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

    def neg_ll(self, y: Sequence, mn: Sequence):

        """
        Calculates the negative log likelihood of outputs given predicted means.

        Args:

            y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
            shape n_smps*d_out[h]

            mns: A sequence of predicted means.  mns[h] contains the predicted means for group h.  mns[h]
            should be of shape n_smps*d_out[h]

        Returns:
            The calculated negative log-likelihood for the sample
        """

        neg_ll = float(0)

        n_smps = y[0].shape[0]
        neg_log_2_pi = float(np.log(2*np.pi))

        for mn_h, y_h, psi_h in zip(mn, y, self.psi):
            neg_ll += .5*mn_h.nelement()*neg_log_2_pi
            neg_ll += .5*n_smps*torch.sum(torch.log(psi_h))
            neg_ll += .5*torch.sum(((y_h - mn_h)**2)/psi_h)

        return neg_ll

    def fit(self, x: Sequence[torch.Tensor], y: Sequence[torch.Tensor],
            batch_size: int=100, send_size: int=100, max_its: int=10,
            learning_rates=.01, adam_params: dict = {}, min_var: float = 0.0, update_int: int = 1000,
            parameters: list = None):

        """ Fits a model to data.

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

                start_end = end_ind
                end_ind = np.min([batch_size, start_end + send_size])

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


class LinearMap(torch.nn.Module):
    """ Wraps torch.nn.Linear for use with a latent mapping. """

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
            y: Output.  y[h] gives the output for output group h as a tensor or shape n_smps*n_dims
        """

        return x

