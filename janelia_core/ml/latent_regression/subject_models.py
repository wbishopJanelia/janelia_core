""" Defines the class for single subject latent-regression models. """

import copy
import itertools
import re
import time
from typing import List, Sequence, Union

import numpy as np
import torch

import warnings

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

    Next, w_h is formed by optionally applying element-wise scales and offsets to o_h. If these scales and offsets are
    not used, w_h = o_h.

    The user can also specify pairs (g, h) when d_in^g = d_out^h, where there is a direct mapping from x_g to a
    vector v_h, v_h = c_{h,g} x_g, where c_{h,g} is a diagonal matrix.  This is most useful when x_g and
    y_g are the same set of variables (e.g, neurons) at times t-1 and t, and in addition to low-rank interactions,
    we want to include interactions between each variable and itself.

    Variables mn_h = w_h + v_h are then formed, and finally, y_h = mn_h + n_h, where n_h ~ N(0, psi_h)
    where psi_h is a diagonal covariance matrix.

    """

    def __init__(self, d_in: Sequence[int], d_out: Sequence[int], d_proj: Sequence[int], d_trans: Sequence[int],
                 m: torch.nn.Module, s: Sequence[torch.nn.Module], use_scales: bool = False,
                 assign_scales: Union[bool, Sequence[bool]] = True, use_offsets: bool = False,
                 assign_offsets: Union[bool, Sequence[bool]] = True, direct_pairs: Sequence[tuple] = None,
                 assign_direct_pair_mappings: Union[bool, Sequence[bool]] = True,
                 assign_p_modes: Union[bool, Sequence[bool]] = True, assign_u_modes: Union[bool, Sequence[bool]] = True,
                 assign_psi: Union[bool, Sequence[bool]] = True, w_gain: float = 1, sc_std: float = .01,
                 dm_std: float = .1, noise_range: Sequence[float] = [.1, .2]):
        """ Create a LatentRegModel object.

        When creating the object, the user has the option to "assign" different variables.  This means a (potentiallY)
        learnable parameter will be created for the variable.  A user may select not to assign a variable if the
        model will be fit with a probabilistic framework where distributions over different parameters will be used
        in place of point estimates.  In this case, because the variables stored with this object will not be used,
        the user can chose to save memory by not creating that variable in the first place.  The user has two different
        ways to specify if variables are assigned.  A user can enter a single boolean value (e.g., assign_psi = True),
        in which case psi variables for all output groups will be assigned.  Alternatively, the user can provide a
        sequence of boolean values, indicating which variables for each group should be assigned.

        Once parameters are created, the user can select if they are learnable or not by manipulating the internal
        trainable parameters for the object.  By default, all created parameters are learnable.  However, a user
        might want to set some parameters by hand and then hold them fixed, in which case setting these parameters
        to not be learnable will be useful.

        Args:

            d_in: d_in[g] gives the input dimensionality for group g of input variables.

            d_out: d_out[h] gives the output dimensionality for group h of output variables.

            d_proj: d_proj[g] gives the dimensionality for the projected latent variables for input group g.

            d_trans: d_trans[h] gives the dimensionality for the transformed latent variables for output group h.

            m: The mapping from [p_1, ..., p_G] to [t_h, ..., t_h].

            s: s[h] contains module to be applied to z_h (see above).

            use_scales: True if scales should be applied to the o_h values of each output group.

            assign_scales: True if scales should be assigned.  See note above on assigning variables.

            use_offsets: True if offsets should be applied to the o_h values of each output group.

            assign_offsets: True if offsets should be assigned.  See note above on assigning variables.

            direct_pairs: direct_pairs[p] contains a tuple of the form (g, h) giving a pair of input and output groups
            that should have direct connections.

            assign_direct_pair_mappings: True if direct pair mappings should be assigned.  See note above on assigning
            variables.  If indicating which particular direct pair mappings should be assigned, this should be a
            sequence and the i^th entry in the sequence indicates if the i^th pair in direct_pairs has a mapping
            assigned.

            assign_p_modes: True if p modes should be assigned.  See note above on assigning variables.

            assign_u_modes: True if u modes should be assigned.  See note above on assigning variables.

            assign_psi: True if psi variables should be assigned.  See note above on assigning variables.

            w_gain: Gain to apply to projection p and u matrices when initializing their weights.

            sc_std: Standard deviation for initializing scale values.

            dm_std: Standard deviation for initializing direct mappings.

            noise_range: Range of uniform distribution to pull psi values from during initialization.

        """

        super().__init__()

        # Record basic parameters
        self.d_in = d_in
        self.d_out = d_out
        self.d_proj = d_proj
        self.d_trans = d_trans
        self.use_scales = use_scales
        self.use_offsets = use_offsets
        self.direct_pairs = direct_pairs
        self.use_direct_pairs = direct_pairs is not None

        n_input_groups = len(d_in)
        self.n_input_groups = n_input_groups
        n_output_groups = len(d_out)
        self.n_output_groups = n_output_groups

        # Put our assignment arguments into a standard form
        if isinstance(assign_p_modes, bool):
            assign_p_modes = [assign_p_modes]*n_input_groups
        if isinstance(assign_u_modes, bool):
            assign_u_modes = [assign_u_modes]*n_output_groups
        if isinstance(assign_scales, bool):
            assign_scales = [assign_scales]*n_output_groups
        if isinstance(assign_offsets, bool):
            assign_offsets = [assign_offsets]*n_output_groups
        if isinstance(assign_direct_pair_mappings, bool) and (direct_pairs is not None):
            assign_direct_pair_mappings = [assign_direct_pair_mappings]*len(direct_pairs)
        if isinstance(assign_psi, bool):
            assign_psi = [assign_psi]*n_output_groups


        # Mapping from projection to transformed latents
        self.m = m

        # Mappings from transformed latents to o_h
        self.s = torch.nn.ModuleList(s)

        # Initialize projection matrices down
        p = [None]*n_input_groups
        for g, dims in enumerate(zip(d_in, d_proj)):
            if assign_p_modes[g]:
                param_name = 'p' + str(g)
                p[g] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
                torch.nn.init.xavier_normal_(p[g], gain=w_gain)
                self.register_parameter(param_name, p[g])
        self.p = p
        self.p_trainable = assign_p_modes  # All assigned projection matrices are by default trainable

        # Initialize projection matrices up
        u = [None]*n_output_groups
        for h, dims in enumerate(zip(d_out, d_trans)):
            if assign_u_modes[h]:
                param_name = 'u' + str(h)
                u[h] = torch.nn.Parameter(torch.zeros([dims[0], dims[1]]), requires_grad=True)
                torch.nn.init.xavier_normal_(u[h], gain=w_gain)
                self.register_parameter(param_name, u[h])
        self.u = u
        self.u_trainable = assign_u_modes  # All assigned projection matrices are by default trainable

        # Direct mappings - if there are none, we set direct_mappings to None
        if direct_pairs is not None:
            n_direct_pairs = len(direct_pairs)
            direct_mappings = [None] * n_direct_pairs
            for pair_i, pair in enumerate(direct_pairs):
                if assign_direct_pair_mappings[pair_i]:
                    c = torch.nn.Parameter(torch.ones(d_in[pair[0]]), requires_grad=True)
                    torch.nn.init.normal_(c, 0, dm_std)
                    param_name = 'c' + str(pair[0]) + '_' + str(pair[1])
                    self.register_parameter(param_name, c)
                    direct_mappings[pair_i] = c
            self.direct_mappings = direct_mappings
            self.direct_mappings_trainable = assign_direct_pair_mappings

        # Scales
        if use_scales:
            scales = [None]*n_output_groups
            for h in range(n_output_groups):
                if assign_scales[h]:
                    sc = torch.nn.Parameter(torch.ones(d_out[h]), requires_grad=True)
                    torch.nn.init.normal_(sc, 0, sc_std)
                    self.register_parameter('sc' + str(h), sc)
                    scales[h] = sc
            self.scales = scales
            self.scales_trainable = assign_scales

        # Offsets
        if use_offsets:
            offsets = [None]*n_output_groups
            for h in range(n_output_groups):
                if assign_offsets[h]:
                    o = torch.nn.Parameter(torch.zeros(d_out[h]), requires_grad=True)
                    self.register_parameter('o' + str(h), o)
                    offsets[h] = o
            self.offsets = offsets
            self.offsets_trainable = assign_offsets

        # Variances on output variables
        psi = [None]*n_output_groups
        for h, d in enumerate(d_out):
            if assign_psi[h]:
                param_name = 'psi' + str(h)
                psi[h] = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
                torch.nn.init.uniform_(psi[h], noise_range[0], noise_range[1])
                self.register_parameter(param_name, psi[h])
        self.psi = psi
        self.psi_trainable = assign_psi

    def forward(self, x: list) -> Sequence:
        """ Computes the predicted mean from the model given input.

        This function assumes all parameters of the model have been assigned.  If this is not the case and you wish
        to provide some of the values for parameters not assigned within the model, see cond_forward().

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        Returns:
            y: A sequence of outputs. y[h] contains the output for group h.  y[h] will be of shape n_smps*d_out[h]
        """

        return self.cond_forward(x)

    def cond_forward(self, x: List[torch.Tensor],
                     p: Union[List[Union[torch.Tensor, None]], None] = None,
                     u: Union[List[Union[torch.Tensor, None]], None] = None,
                     scales: Union[List[Union[torch.Tensor, None]], None] = None,
                     offsets: Union[List[Union[torch.Tensor, None]], None] = None,
                     direct_mappings: Union[List[Union[torch.Tensor, None]], None] = None):
        """ Computes means given x and different parameter values.

        The user can specify parameter values to override (see arguments below).  When any of these are provided,
        the internal values for this parameter stored with the model are ignored and the user provided values are
        used instead. The user can provide this specification in two ways.  Providing a None value (e.g., p = None),
        specifies all of the paramters for all groups should be used (in this example, p modes in the model for all
        groups would be used).  Alternatively, the user can provide a sequence (e.g, p = [None, t]).  An entry of None
        in the sequence indicates the model's parameter for that group should be used, while if an entry is a tensor,
        than that tensor will be used in place of the model's parameters.

        Args:

            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

            p: Values for the p-modes to use.  See note above on how parameters can be specified.

            u: Values for the u-modes to use.  See note above on how parameters can be specified.

            scales: Values for scales to use.  See note above on how parameters can be specified.

            offsets: Values for offsets to use.  See note above on how parameters can be specified.

            direct_mappings: Direct mappings to use.  When specifying a sequence. direct_mappings[i] contains the values
            for the direct mappings in self.direct_pairs[i]

        Returns:
            y: A sequence of outputs. y[h] contains the means for group h.  y[h] will be of shape n_smps*d_out[h]

        """

        if p is None:
            p = [None]*self.n_input_groups
        if u is None:
            u = [None]*self.n_output_groups
        if self.use_scales and (scales is None):
            scales = [None]*self.n_output_groups
        if self.use_offsets and (offsets is None):
            offsets = [None]*self.n_output_groups
        if self.use_direct_pairs and (direct_mappings is None):
            direct_mappings = [None]*self.n_output_groups

        # Now we pull parameters from the model we are using
        p = [p[g] if p[g] is not None else self.p[g] for g in range(self.n_input_groups)]
        u = [u[h] if u[h] is not None else self.u[h] for h in range(self.n_output_groups)]

        if self.use_scales:
            scales = [scales[h] if scales[h] is not None else self.scales[h] for h in range(self.n_output_groups)]
        if self.use_offsets:
            offsets = [offsets[h] if offsets[h] is not None else self.offsets[h] for h in range(self.n_output_groups)]

        if self.use_direct_pairs:
            direct_mappings = [direct_mappings[i] if direct_mappings[i] is not None
                               else self.direct_mappings[i] for i in range(len(self.direct_mappings))]

        # Compute output
        proj = [torch.matmul(x_g, p_g) for x_g, p_g in zip(x, p)]

        tran = self.m(proj)
        z = [torch.matmul(t_h, u_h.t()) for t_h, u_h in zip(tran, u)]

        o = [s_h(z_h) for s_h, z_h in zip(self.s, z)]

        if self.use_scales:
            w = [sc_h*o_h for sc_h, o_h in zip(scales, o)]
        else:
            w = o

        if self.use_offsets:
            w = [off_h + w_h for off_h, w_h in zip(offsets, w)]

        # Add direct mappings
        if self.use_direct_pairs:
            for pair_i, pair in enumerate(self.direct_pairs):
                g = pair[0]
                h = pair[1]
                w[h] = w[h] + direct_mappings[pair_i]*x[g]

        return w

    def generate(self, x: Sequence) -> Sequence:
        """ Generates outputs from the model given inputs.

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        Returns:
            y: A sequence of generated outputs.  y[h] contains the output tensor for group h.  y[h] will be of
            shape n_smps*d_out[h]
        """

        with torch.no_grad():
            mns = self(x)
            y = [None]*self.n_output_groups
            for h in range(self.n_output_groups):
                noise_h = torch.randn_like(mns[h])*torch.sqrt(self.psi[h])
                y[h] = mns[h] + noise_h

        return y

    def recursive_generate(self, x: Sequence, r_map: list = None) -> Sequence:
        """ Recursively generates output for a given number of time steps.

        The concept behind this function is that to simulate T samples from the model, we specify
        T sets of initial conditions (one set for each sample).  If the initial conditions are
        fully specified (no NAN values) then this function is equivalent to generate(). However,
        if some initial conditions are left unspecified for certain time points, then the output
        of the model from the previous time steps will be used as the initial conditions.
        This gives users the flexibility of simulating scenarios where variables in a model may
        be selectively "clamped" at different points in time.

        Args:
            x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] is
            of shape [n_smps, d_x], where n_smps is the number of samples we simulate
            output for.  Specifically, x[g][t, :] contains the initial conditions for group g and
            time point t.  Any nan values indicate the initial conditions for that variable should
            be pulled from the output of the model from the previous time step.

            r_map: Specifies which output groups should be mapped to which input groups when
            recursively generating data.  Any input group which does not have output mapped to
            must have all values in it's entry in x fully specified.  r_map[i] is a tuple of
            the form (h,g) specifying that the output of group h should be recursively mapped
            to the input of group g.

        Returns:
            y: A sequence of outputs.  y[h] contains the output for group h.  Will be of shape n_smps*d_y.

        Raises:
            ValueError: If any initial conditions contain nan values.

            ValueError: If any input group which does not recursively receive output does not
            have all of its values in x specified.

        """

        # Make sure initial conditions are fully specified
        for x_g in x:
            if torch.any(torch.isnan(x_g[0, :])):
                raise(ValueError('First row of each x tensor must not contain any nan values.'))

        # Make sure any input groups which are not recursively generated have all input values specified
        n_input_grps = len(self.d_in)
        mapped_input_grps = set([m_i[1] for m_i in r_map])
        unmapped_input_grps = set(range(n_input_grps)).difference(mapped_input_grps)
        for g in unmapped_input_grps:
            if torch.any(torch.isnan(x[g])):
                raise(ValueError('The x tensors for all input groups which do not receive recursively generated ' +
                                  ' output must contain no nan values.'))

        # Recursively generate output
        n_smps = x[0].shape[0]
        y = [torch.zeros([n_smps, d]) + np.nan for d in self.d_out]  # Add nan to see if we failed to assign values

        x = copy.deepcopy(x)
        for i in range(n_smps):
            cur_x = [x_g[None, i, :] for x_g in x]

            # Fill in the values of nan values here
            for h, g in r_map:
                nan_vars = torch.nonzero(torch.isnan(cur_x[g])).squeeze()
                cur_x[g][0, nan_vars] = y[h][i-1, nan_vars]

            output = self.generate(cur_x)
            for h, o_h in enumerate(output):
                y[h][i, :] = o_h
        return y

    def s_parameters(self):
        """ Gets the parameters of the s modules.

        Returns:
            params: params[i] is a list of parameters for the i^th output group
        """
        return itertools.chain(*[s.parameters() for s in self.s])

    def neg_ll(self, y: Sequence[torch.Tensor], mn: Sequence[torch.Tensor],
               psi: Sequence[torch.Tensor] = None) -> torch.Tensor:

        """
        Calculates the negative log likelihood of outputs given predicted means.

        Args:

            y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
            shape n_smps*d_out[h]

            mn: A sequence of predicted means.  mns[h] contains the predicted means for group h.  mns[h]
            should be of shape n_smps*d_out[h]

            psi: An optional value of psi to use.  Can be specified in two ways.  If None, then the model's internal
            parameter for psi for all groups will be used.  If a sequence, then if psi[h] is None, the model's
            parameter of psi[h] will be used.  However is psi[h] is a tensor, then this value will be used in place
            of the models.

        Returns:
            The calculated negative log-likelihood for the sample

        Raises:
            ValueErorr: If y and mn are not lists
        """

        if not isinstance(y, list):
            raise(ValueError('y must be a list'))
        if not isinstance(mn, list):
            raise(ValueError('mn must be a list'))

        # Put psi argument in standard form
        if psi is None:
            psi = [None]*self.n_output_groups

        # Overwrite psi values for particular groups if we need to
        psi = [psi[h] if psi[h] is not None else self.psi[h] for h in range(self.n_output_groups)]

        # Calculate negative log-likelihood
        neg_ll = float(0)
        n_smps = y[0].shape[0]
        log_2_pi = float(np.log(2*np.pi))
        for mn_h, y_h, psi_h, in zip(mn, y, psi):
            neg_ll += .5*mn_h.nelement()*log_2_pi
            neg_ll += .5*n_smps*torch.sum(torch.log(psi_h))
            neg_ll += .5*torch.sum(((y_h - mn_h)**2)/psi_h)

        return neg_ll

    def trainable_parameters(self) -> List[torch.nn.parameter.Parameter]:
        """ Gets all trainable parameters of the model.

        Trainable parameters are those in the s and m modules as well as the p modes, u modes, scale, offset, psi and
        direct_mapping parameters which are set to trainable (e.g., self.p_trainable has true entries for the
        groups with trainable p modes).
        """

        m_params = self.m.parameters()
        s_params = self.s_parameters()

        p_params = [self.p[g] for g, trainable in enumerate(self.p_trainable) if trainable and self.p[g] is not None]
        u_params = [self.u[h] for h, trainable in enumerate(self.u_trainable) if trainable and self.u[h] is not None]

        if self.use_scales:
            scale_params = [self.scales[h] for h, trainable in enumerate(self.scales_trainable) if trainable
                            and self.scales[h] is not None]
        else:
            scale_params = []

        if self.use_offsets:
            offset_params = [self.offsets[h] for h, trainable in enumerate(self.offsets_trainable) if trainable
                             and self.offsets[h] is not None]
        else:
            offset_params = []

        if self.use_direct_pairs:
            direct_mapping_params = [self.direct_mappings[i] for i, trainable in
                                     enumerate(self.direct_mappings_trainable) if trainable
                                     and self.direct_mappings[i] is not None]
        else:
            direct_mapping_params = []

        psi_params = [self.psi[h] for h, trainable in enumerate(self.psi_trainable) if trainable
                      and self.psi[h] is not None]

        return list(itertools.chain(m_params, s_params, p_params, u_params, scale_params, offset_params,
                                    direct_mapping_params, psi_params))

    def fit(self, x: Sequence[torch.Tensor], y: Sequence[torch.Tensor],
            batch_size: int=100, send_size: int=100, max_its: int=10,
            learning_rates=.01, adam_params: dict = {}, min_var: float = 0.000001, update_int: int = 1000,
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
            parameters = self.trainable_parameters()
        # Convert generator to list (since we need to reference parameters multiple times in the code below)
        parameters = [p for p in parameters]

        # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
        # where the learning rate starting at iteration 0 is guaranteed to be listed first
        learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

        optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0,0], **adam_params)

        n_smps = x[0].shape[0]
        cur_it = 0
        start_time = time.time()

        elapsed_time_log = np.zeros(max_its)
        obj_log = np.zeros(max_its)
        prev_learning_rate = learning_rate_values[0, 0]

        while cur_it < max_its:
            elapsed_time = time.time() - start_time  # Record elapsed time here because we measure it from the start of
            # each iteration.  This is because we also record the nll value for each iteration before parameters are
            # updated.  In this way, the elapsed time is the elapsed time to get to a set of parameters for which we
            # report the nll.

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= cur_it)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind, 0]
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


class SharedMLatentRegModel(LatentRegModel):
    """ A base class for latent regression models with m-modules that are at least partially shared between instances.

    This class is designed to ease working in scenarios where we fit multiple latent regression models, one to each
    individual, and we want the m-module of these models to have a component that is shared across all of these models.

    The m-module of these instances will be of the form m = torch.nn.Sequential(specific_m, shared_m), where specific_m
    is an module unique to the instance and shared_m is a module shared between instances.

    This class provides convenience methods that facilitate getting the shared and specific portions and parameters of
    the m-module.

    """

    def __init__(self, d_in: Sequence, d_out: Sequence, d_proj: Sequence, d_trans: Sequence,
                 specific_m: torch.nn.Module, shared_m: torch.nn.Module, s: Sequence[torch.nn.Module],
                 use_scales: bool = False, assign_scales: Union[bool, Sequence[bool]] = True, use_offsets: bool = False,
                 assign_offsets: Union[bool, Sequence[bool]] = True, direct_pairs: Sequence[tuple] = None,
                 assign_direct_pair_mappings: Union[bool, Sequence[bool]] = True,
                 assign_p_modes: Union[bool, Sequence[bool]] = True, assign_u_modes: Union[bool, Sequence[bool]] = True,
                 assign_psi: Union[bool, Sequence[bool]] = True, w_gain: float = 1, sc_std: float = .01,
                 dm_std: float = .1, noise_range: Sequence[float] = [.1, .2]):

        if (specific_m is not None) and (shared_m is not None):
            m = torch.nn.Sequential(specific_m, shared_m)
        elif specific_m is None:
            m = shared_m
        else:
            m = specific_m

        super().__init__(d_in=d_in, d_out=d_out, d_proj=d_proj, d_trans=d_trans, m=m, s=s, use_scales=use_scales,
                         assign_scales=assign_scales, use_offsets=use_offsets, assign_offsets=assign_offsets,
                         direct_pairs=direct_pairs, assign_direct_pair_mappings=assign_direct_pair_mappings,
                         assign_p_modes=assign_p_modes, assign_u_modes=assign_u_modes, assign_psi=assign_psi,
                         w_gain=w_gain, sc_std=sc_std, dm_std=dm_std, noise_range=noise_range)

        self.specific_m = specific_m
        self.shared_m = shared_m
        self.return_shared_m_params = True

    def update_shared_m(self, new_m_core: torch.nn.Module):
        """ Updates the shared component of the m-module. """
        self.shared_m = new_m_core
        self.m[1] = new_m_core

    def specific_m_parameters(self) -> list:
        """ Returns subject-specific parameters of the m-module.

        Returns:
            params: A list of parameters.
        """
        if self.specific_m is not None:
            return list(self.specific_m.parameters())
        else:
            return []

    def shared_m_parameters(self) -> list:
        """ Returns shared parameters of the m-module.

        Returns:
            params: A list of parameters.
        """
        if self.shared_m is not None:
            return list(self.shared_m.parameters())
        else:
            return []

    def parameters(self): #-> Generator[torch.nn.Parameter]:
        """ Returns parameters of the model, possibly excluding parameters of the shared component of the m-module.

        The parameters of the shared component of the m-module will not be returned if self.returned_shared_m_params is
        False.

        """
        full_params = list(super().parameters())

        if self.return_shared_m_params:
            return_params = full_params
        else:
            shared_m_params = set(self.shared_m.parameters())
            full_params = set(full_params)
            return_params = list(full_params.difference(shared_m_params))

        return (p for p in return_params)

