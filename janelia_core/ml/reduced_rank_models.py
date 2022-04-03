""" Contains classes and functions for performing different forms of nonlinear reduced
rank regression.

"""

import copy
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import torch
from typing import List

from janelia_core.visualization.matrix_visualization import cmp_n_mats
from janelia_core.ml.utils import format_and_check_learning_rates


class RRLinearModel(torch.nn.Module):
    """ Object for linear reduced rank linear model objects.

    Base class for non-linear reduced rank linear model objects.

    This object and all objects derived from it are for models of the form:

        y_t = g(w_1*w_0^T * x_t + o_1) + o_2 + n_t,

        n_t ~ N(0, V), for a diagonal V

    where x_t is the input and g() is the element-wise application of a smooth function.  In this base class, we fix
    o_1 = 0, g = I.

    Often w_0 and w_1 will be tall and skinny matrices (giving the reduced rank).

    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        """ Create a RRLinearModel object.

        Args:
            d_in: Input dimensionality

            d_out: Output dimensionality

            d_latent: Latent dimensionality
        """
        super().__init__()

        w0 = torch.nn.Parameter(torch.zeros([d_in, d_latent]), requires_grad=True)
        self.register_parameter('w0', w0)

        w1 = torch.nn.Parameter(torch.zeros([d_out, d_latent]), requires_grad=True)
        self.register_parameter('w1', w1)

        o2 = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('o2', o2)

        # v is the *variances* of each noise term
        v = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('v', v)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        """ Creates a new RRReluModel model from a dictionary.

        Args:
            state_dict: The state dictionary to create the model from.  This can be obtained
            by calling .state_dict() on a model.

        Returns:
            A new model with parameters taking values in the state_dict()

        """
        d_in = state_dict['w0'].shape[0]
        d_out = state_dict['w1'].shape[0]
        d_latent = state_dict['w0'].shape[1]

        mdl = cls(d_in, d_out, d_latent)
        mdl.load_state_dict(state_dict)
        return mdl

    def init_weights(self, y: torch.Tensor):
        """ Randomly initializes all model parameters based on data.

        This function should be called before model fitting.

        Args:
            y: Output data that the model will be fit to of shape n_smps*d_out

        Raise:
            NotImplementedError: If a parameter of the model exists for which initialization code
            does not exist.
        """

        d_in = self.w0.shape[0]
        d_out = self.w1.shape[0]

        var_variance = np.reshape(np.var(y.numpy(), 0), [d_out, 1])
        var_mean = np.reshape(np.mean(y.numpy(), 0), [d_out, 1])

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data = torch.from_numpy(var_variance)
            elif param_name in {'o2'}:
                param.data = torch.from_numpy(var_mean)
            elif param_name in {'w0'}:
                param.data.normal_(0, 1/np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, 1)
            else:
                raise(NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def generate_random_model(self, var_range: list = [.5, 1], o2_std: float = 10.0, w_gain: float = 1.0):
        """ Generates random values for model parameters.

        This function is useful for when generating models for testing code.

        Args:
            var_range: A list giving limits of a uniform distribution variance values will be pulled from

            o2_std: Standard deviation of normal distribution values of o2 are pulled from

            w_gain: Entries of w0 and w1 are pulled from a distribution with a standard deviation of w_gain/sqrt(d_in),
            and entries of w1 are pulled from a distribution with a standard deviation of w_gain

        """

        d_in = self.w0.shape[0]

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data.uniform_(*var_range)
            elif param_name in {'o2'}:
                param.data.normal_(0, o2_std)
            elif param_name in {'w0'}:
                param.data.normal_(0, w_gain / np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, w_gain)
            else:
                raise (NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output means condition on x.

        This is equivalent to running the full generative model but not adding noise from n_t.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            mns: The output.  Of shape n_smps*d_out

        """
        x = torch.matmul(torch.t(self.w0), torch.t(x))
        x = torch.matmul(self.w1, x)
        x = x + self.o2
        return torch.t(x)

    def infer_latents(self, x: torch.Tensor) -> torch.Tensor:
        """ Infers latents defined as w0.T*x.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            l: Inferred latents of shape n_smps*d_latents.

        """
        with torch.no_grad():
            return torch.matmul(x, self.w0)

    def scale_grads(self, sc: float):
        """ Scales computed gradients.

        Args:
            sc: The scale factor to multiply gradients by.

        """
        for param in self.parameters():
            if param.requires_grad:
                param.grad.data = sc*param.grad.data

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """ Generates data from a model.

        This is equivalent to calling forward() and then adding noise from n_t.

        Note: This function will not effect gradients.

        Args:
            x: Input of shape n_smps*n_dims

        Returns:
            y: The output.  Same shape as x.
        """
        with torch.no_grad():
            mns = self.forward(x)
            noise = torch.randn_like(mns)*torch.sqrt(torch.t(self.v))
            return mns + noise

    def neg_log_likelihood(self, y: torch.Tensor, mns: torch.Tensor) -> torch.Tensor:

        """ Calculates the negative log-likelihood of observed data, given conditional means for that data up to a constant.

        Note: This function does not compute the term .5*n_smps*log(2*pi).  Add this term in if you want the exact
        log likelihood.

        This function can be used as a loss, using the output of forward for mns and setting y to be observed data.
        Using this function as loss will give (subject to local optima) MLE solutions.

        Args:
            y: Data to measure the negative log likelihood for of shape n_smps*d_in.

            mns: The conditional means for the data.  These can be obtained with forward().

        Returns:
            nll: The negative log-likelihood of the observed data.

        """

        n_smps = y.shape[0]
        return .5 * n_smps*torch.sum(torch.log(self.v)) + .5 * torch.sum(((y - mns) ** 2) / torch.t(self.v))

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int=100, send_size: int=100, max_its: int=10,
            learning_rates=.01, adam_params: dict = {}, min_var: float = 0.0, update_int: int = 1000,
            parameters: list = None, w0_l2: float = 0.0, w1_l2: float = 0, w0_l1: float = 0, w1_l1: float = 0,
            print_penalties: bool = False) -> dict:
        """ Fits a model to data.

        This function performs stochastic optimization with the ADAM algorithm.  The weights of the model
        should be initialized before calling this function.

        Optimization will be performed on whatever device the model parameters are on.

        Args:

            x: Tensor of input data of shape n_smps*d_in

            y: Tensor of output data of shape n_smps*d_out

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

            min_var: The minumum value any entry of v can take on.  After a gradient update, values less than this
            will be clamped to this value.

            update_int: The interval of iterations we update the user on.

            parameters: If provided, only these parameters of the model will be optimized.  If none, all parameters are
            optimized.

            w0_l2: The penalty on the l-2 norm of the model's w0 weights.

            w1_l2: The penalty on the l-2 norm of the model's w1 weights.

            w0_l1: The penalty on the l-1 norm of the model's w0 weights.

            w1_l1: The penalty on the l-2 norm of the model's w1 weights.

            print_penalties: True if penalty values should be printed to screen.

        Raises:
            ValueError: If send_size is greater than batch_size.

        Returns:
            log: A dictionary logging progress.  Will have the enries:
                'elapsed_time': log['elapsed_time'][i] contains the elapsed time from the beginning of optimization to
                the end of iteration i

                'obj': log['obj'][i] contains the objective value at the beginning (before parameters are updated) of iteration i.

            """

        if send_size > batch_size:
            raise(ValueError('send_size must be less than or equal to batch_size.'))

        device = self.w0.device

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

        n_smps = x.shape[0]
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

            # Chose the samples for this iteration
            cur_smps = np.random.choice(n_smps, batch_size, replace=False)
            batch_x = x[cur_smps, :]
            batch_y = y[cur_smps, :]

            # Perform optimization for this step
            optimizer.zero_grad()

            # Handle sending data to device in small chunks if needed
            start_ind = 0
            end_ind = np.min([batch_size, send_size])
            while True:
                sent_x = batch_x[start_ind:end_ind, :].to(device)
                sent_y = batch_y[start_ind:end_ind, :].to(device)

                mns = self(sent_x.data)
                # Calculate nll - we divide by batch size to get average (over samples) negative log-likelihood
                obj = (1/batch_size)*self.neg_log_likelihood(sent_y.data, mns)
                obj.backward()

                if end_ind == batch_size:
                    break

                start_end = end_ind
                end_ind = np.min([batch_size, start_end + send_size])

            # Add penalties to average negative log-likelihood if needed
            if w0_l2 > 0:
                w0_l2_penalty = w0_l2 * (self.w0 * self.w0).sum()
                w0_l2_penalty.backward() # Apply backwards just to penalty terms since gradients accumulate and we
                                         # we have already done this for the likelihood
                obj = obj + w0_l2_penalty

            if w1_l2 > 0:
                w1_l2_penalty = w1_l2 * (self.w1 * self.w1).sum()
                w1_l2_penalty.backward()
                obj = obj + w1_l2_penalty

            if w0_l1 > 0:
                w0_l1_penalty = w0_l1*(torch.abs(self.w0)).sum()
                w0_l1_penalty.backward()
                obj = obj + w0_l1_penalty

            if w1_l1 > 0:
                w1_l1_penalty = w1_l1*(torch.abs(self.w1)).sum()
                w1_l1_penalty.backward()
                obj = obj + w1_l1_penalty

            # Take a step
            optimizer.step()

            with torch.no_grad():
                small_v_inds = torch.nonzero(self.v < min_var)
                self.v.data[small_v_inds] = min_var

            # Log our progress
            elapsed_time_log[cur_it] = elapsed_time
            obj_vl = obj.cpu().detach().numpy()
            obj_log[cur_it] = obj_vl

            # Provide user with some feedback
            if cur_it % update_int == 0:
                print(str(cur_it) + ': Elapsed fitting time ' + str(elapsed_time) +
                      ', vl: ' + str(obj_vl) + ', lr: ' + str(cur_learning_rate))
                if print_penalties:
                    if w0_l2 > 0:
                        print('w0 l-2 penalty:  ' + str(w0_l2_penalty.cpu().detach().numpy()))
                if print_penalties:
                    if w1_l2 > 0:
                        print('w1 l-2 penalty:  ' + str(w1_l2_penalty.cpu().detach().numpy()))
                if print_penalties:
                    if w0_l1 > 0:
                        print('w0 l-1 penalty:  ' + str(w0_l1_penalty.cpu().detach().numpy()))
                if print_penalties:
                    if w1_l1 > 0:
                        print('w1 l-1 penalty:  ' + str(w1_l1_penalty.cpu().detach().numpy()))

            cur_it += 1

        # Give final fitting results (if we have not already)
        if update_int != 1:
            print(str(cur_it-1) + ': Elapsed fitting time ' + str(elapsed_time) +
                  ', vl: ' + str(obj_vl))

        log = {'elapsed_time': elapsed_time_log, 'obj': obj_log}

        return log

    def standardize(self):
        """ Puts the model in a standard form.

            The values of w0 and w1 are not fully determined.  The svd of w0*w1.T = u*s*v.T will be performed,
            and then w0 will be set to u*s and w1 will be set to v.
        """
        latent_dim = self.w0.shape[1]
        output_dim = self.w1.shape[0]

        w0 = self.w0.cpu().detach().numpy()
        w1 = self.w1.cpu().detach().numpy()
        full_w_t = np.matmul(w0, w1.T)

        if latent_dim < output_dim:
            # Truncated svd only works for latent_dim < output_dim
            svd = sklearn.decomposition.TruncatedSVD(latent_dim)
            svd.fit(full_w_t)
            v = svd.components_.T
            u_s = svd.transform(full_w_t)

        else:
            u, s, v_t = np.linalg.svd(full_w_t)
            v = v_t.T
            u_s = np.matmul(u, np.diag(s))

        self.w0.data = torch.from_numpy(u_s).to(device=self.w0.device, dtype=self.w0.dtype)
        self.w1.data = torch.from_numpy(v).to(device=self.w1.device, dtype=self.w1.dtype)

    def flip_weight_signs(self, m2):
        """ Flips signs of weights, column-wise of input model to best match weights of this model.

        Even after standardizing the weights of the model, signs of elements in columns of w0 and w1 are not
        determined.  This creates a problem when comparing two models.  This function can be passed a second model,
        and it will flip the signs of the columns of that model's weights to best match the weights of the base
        model.

        Args:
            m2: The model with weights to flip
        """

        n_dims = self.w1.shape[1]
        for d_i in range(n_dims):
            e1 = ((self.w0[:, d_i] - m2.w0[:, d_i]) ** 2).sum() + ((self.w1[:, d_i] - m2.w1[:, d_i]) ** 2).sum()
            e2 = ((self.w0[:, d_i] + m2.w0[:, d_i]) ** 2).sum() + ((self.w1[:, d_i] + m2.w1[:, d_i]) ** 2).sum()
            if e2 < e1:
                print('Flipping signs for dimension ' + str(d_i))
                m2.w0[:, d_i] = -1 * m2.w0[:, d_i]
                m2.w1[:, d_i] = -1 * m2.w1[:, d_i]

    @staticmethod
    def compare_models(m1, m2, x: torch.Tensor = None, plot_vars: int = 2):
        """ Visually compares two models.

        This function will flip signs on columns of w0 and w1 to best match weights between models (as signs of
        corresponding columns of w0 and w1 can be flipped arbitrarily).  Models will be copied before this is done
        so the m1 and m2 passed in are unchanged.

        Args:
            m1: The fist model

            m2: The second model

            x: Input to test the models on.  The conditional means for both model will be plotted for this input if
            provided.

            plot_vars: Indices of variables to plot if plotting conditional means. If this None, the first (up to) two
            variables will be plotted.
        """

        ROW_SPAN = 14  # Number of rows in the gridspec
        COL_SPAN = 12  # Number of columns in the gridspec

        m2 = copy.deepcopy(m2) # Copy m2 to local variable so changes to weights don't affect the passed object

        grid_spec = matplotlib.gridspec.GridSpec(ROW_SPAN, COL_SPAN)

        def _make_subplot(loc, r_span, c_span, d1, d2, title):
            subplot = plt.subplot(grid_spec.new_subplotspec(loc, r_span, c_span))
            subplot.plot(d1, 'b-')
            subplot.plot(d2, 'r-')
            subplot.title = plt.title(title)

        def _flip_signs(m1_w0, m2_w0, m1_w1, m2_w1):
            n_dims = m1_w1.shape[1]
            for d_i in range(n_dims):
                e1 = np.sum((m1_w0[:, d_i] - m2_w0[:, d_i])**2) + np.sum((m1_w1[:, d_i] - m2_w1[:, d_i])**2)
                e2 = np.sum((m1_w0[:, d_i] + m2_w0[:, d_i])**2) + np.sum((m1_w1[:, d_i] + m2_w1[:, d_i])**2)
                if e2 < e1:
                    print('Flipping signs for dimension ' + str(d_i))
                    m2_w0[:, d_i] = -1*m2_w0[:, d_i]
                    m2_w1[:, d_i] = -1*m2_w1[:, d_i]
            return [m1_w0, m2_w0, m1_w1, m2_w1]

        # Make plots of scalar variables
        if hasattr(m1, 'g'):
            _make_subplot([0, 0], 2, 2, m1.g.cpu().detach().numpy(), m2.g.cpu().detach().numpy(), 'g')
        if hasattr(m1, 'o2'):
            _make_subplot([3, 0], 2, 2, m1.o2.cpu().detach().numpy(), m2.o2.cpu().detach().numpy(), 'o2')
        if hasattr(m1, 'o1'):
            _make_subplot([6, 0], 2, 2, m1.o1.cpu().detach().numpy(), m2.o1.cpu().detach().numpy(), 'o1')
        if hasattr(m1, 'v'):
            _make_subplot([9, 0], 2, 2, m1.v.cpu().detach().numpy(), m2.v.cpu().detach().numpy(), 'v')

        # Flip signs of weight matrices if needed
        m1.flip_weight_signs(m2)
        m1_w0 = m1.w0.cpu().detach().numpy().T # Transpose w0 for viewing
        m2_w0 = m2.w0.cpu().detach().numpy().T
        m1_w1 = m1.w1.cpu().detach().numpy()
        m2_w1 = m2.w1.cpu().detach().numpy()

        # Make plots of w1 matrices
        w1_diff = m1_w1 - m2_w1
        w1_grid_info = {'grid_spec': grid_spec}
        w1_cell_info = list()
        w1_cell_info.append({'loc': [0, 2], 'rowspan': 10, 'colspan': 1})
        w1_cell_info.append({'loc': [0, 4], 'rowspan': 10, 'colspan': 1})
        w1_cell_info.append({'loc': [0, 6], 'rowspan': 10, 'colspan': 1})
        w1_grid_info['cell_info'] = w1_cell_info
        cmp_n_mats([m1_w1, m2_w1, w1_diff], show_colorbars=True, titles=['m1_w1', 'm2_w1', 'm1_w1 - m2_w1'],
                   grid_info=w1_grid_info)

        # Make plots of w0 matrices

        w0_diff = m1_w0 - m2_w0
        w0_grid_info = {'grid_spec': grid_spec}
        w0_cell_info = list()
        w0_cell_info.append({'loc': [0, 8], 'rowspan': 1, 'colspan': COL_SPAN - 8 + 1})
        w0_cell_info.append({'loc': [2, 8], 'rowspan': 1, 'colspan': COL_SPAN - 8 + 1})
        w0_cell_info.append({'loc': [4, 8], 'rowspan': 1, 'colspan': COL_SPAN - 8 + 1})
        w0_grid_info['cell_info'] = w0_cell_info
        cmp_n_mats([m1_w0, m2_w0, w0_diff], show_colorbars=True, titles=['m1_w0', 'm2_w0', 'm1_w0 - m2_w0'],
                   grid_info=w0_grid_info)

        # Make plot of model output
        if x is not None:
            m1_mns = m1(x).detach().numpy()
            m2_mns = m2(x).detach().numpy()
            mns_plot = plt.subplot(grid_spec.new_subplotspec([12, 0], 2, COL_SPAN))

            if plot_vars is None:
                n_vars = m1_mns.shape[1]
                plot_vars = range(0, np.min([2, n_vars]))

            if len(plot_vars) != 0:
                for v_i in plot_vars:
                    v1_plot = mns_plot.plot(m1_mns[:, v_i])
                    v2_plot = mns_plot.plot(m2_mns[:, v_i], 'o')
                    v2_plot[0].set_color(v1_plot[0].get_color())


class RRSigmoidModel(RRLinearModel):
    """ Sigmoidal non-linear reduced rank model.

    For models of the form:

        y_t = sig(w_1*w_0^T * x_t + o_1) + o_2 + n_t,

        n_t ~ N(0, V), for a diagonal V

    where x_t is the input and sig() is the element-wise application of the sigmoid.  We assume w_0 and w_1 are
    tall and skinny matrices.

    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        """ Create a RRSigmoidModel object.

        Args:
            d_in: Input dimensionality

            d_out: Output dimensionality

            d_latent: Latent dimensionality
        """
        super().__init__(d_in, d_out, d_latent)

        o1 = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('o1', o1)

        g = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('g', g)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        """ Creates a new RRSigmoidModel model from a dictionary.

        Args:
            state_dict: The state dictionary to create the model from.  This can be obtained
            by calling .state_dict() on a model.

        Returns:
            A new model with parameters taking values in the state_dict()

        """
        d_in = state_dict['w0'].shape[0]
        d_out = state_dict['w1'].shape[0]
        d_latent = state_dict['w0'].shape[1]

        mdl = cls(d_in, d_out, d_latent)
        mdl.load_state_dict(state_dict)
        return mdl

    def init_weights(self, y: torch.Tensor):
        """ Randomly initializes all model parameters based on data.

        This function should be called before model fitting.

        Args:
            y: Output data that the model will be fit to of shape n_smps*d_out

        Raise:
            NotImplementedError: If a parameter of the model exists for which initialization code
            does not exist.
        """

        d_in = self.w0.shape[0]
        d_out = self.w1.shape[0]

        var_variance = np.reshape(np.var(y.numpy(), 0), [d_out, 1])
        var_mean = np.reshape(np.mean(y.numpy(), 0), [d_out,1])

        for param_name, param in self.named_parameters():
            if param_name in {'v', 'g'}:
                param.data = torch.from_numpy(var_variance/2)
            elif param_name in {'o2'}:
                param.data = torch.from_numpy(var_mean)
            elif param_name in {'o1'}:
                param.data = torch.zeros_like(param.data)
            elif param_name in {'w0'}:
                param.data.normal_(0, 1/np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, 1)
            else:
                raise(NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def generate_random_model(self, var_range: list = [.5, 1], g_range: list = [5, 10],
                              o1_range: list = [-.2, .2], o2_range: list = [5, 10],
                              w_gain: float = 1.0):
        """ Genarates random values for model parameters.

        This function is useful for when generating models for testing code.

        Args:
            var_range: A list giving limits of a uniform distribution variance values will be pulled from

            g_range: A list giving limits of a uniform distribution g values will be pulled from

            o1_range: A list giving limits of a uniform distribution o1 values will be pulled from

            o2_range: A list giving limits of a uniform distribution o2 values will be pulled from

            w_gain: Entries of w0 and w1 are pulled from a distribution with a standard deviation of w_gain/sqrt(d_in),
            and entries of w1 are pulled from a distribution with a standard deviation of w_gain

        """

        d_in = self.w0.shape[0]

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data.uniform_(*var_range)
            elif param_name in {'g'}:
                param.data.uniform_(*g_range)
            elif param_name in {'o1'}:
                param.data.uniform_(*o1_range)
            elif param_name in {'o2'}:
                param.data.uniform_(*o2_range)
            elif param_name in {'w0'}:
                param.data.normal_(0, w_gain / np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, w_gain)
            else:
                raise (NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output means condition on x.

        This is equivalent to running the full generative model but not adding noise from n_t.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            mns: The output.  Of shape n_smps*d_out

        """
        x = torch.matmul(torch.t(self.w0), torch.t(x))
        x = torch.matmul(self.w1, x)
        x = x + self.o1
        x = torch.sigmoid(x)
        x = self.g*x
        x = x + self.o2
        return torch.t(x)

    def standardize(self):
        """ Puts the model in a standard form.

        The models have multiple degenerecies (non-identifiabilities):

            1) The signs of gains (g) is arbitrary.  A change in sign can be absorbed by a change
            in w1, o1 and d.  This function will put models in a form where all gains have positive
            sign.

            2) Even after (1), the values of w0 and w1 are not fully determined.
            See RRLinearModel.standardize() for how this is done.
        """

        # Standardize with respect to gains
        for i in range(len(self.g)):
            if self.g[i] < 0:
                self.w1.data[i, :] = -1*self.w1.data[i, :]
                self.o1.data[i] = -1*self.o1.data[i]
                self.o2.data[i] = self.o2.data[i] + self.g.data[i]
                self.g.data[i] = -1*self.g.data[i]

        # Standardize with respect to weights
        super().standardize()


class RRExpModel(RRLinearModel):
    """ Exponential non-linear reduced rank model.

    For models of the form:

        y_t = exp(w_1*w_0^T * x_t) + o_2 + n_t,

        n_t ~ N(0, V), for a diagonal V

    where x_t is the input and exp() is the element-wise application of e^x.  We assume w_0 and w_1 are
    tall and skinny matrices.

    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        """ Create a RRExpModel object.

        Args:
            d_in: Input dimensionality

            d_out: Output dimensionality

            d_latent: Latent dimensionality
        """
        super().__init__(d_in, d_out, d_latent)

        g = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('g', g)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        """ Creates a new RRExpModel model from a dictionary.

        Args:
            state_dict: The state dictionary to create the model from.  This can be obtained
            by calling .state_dict() on a model.

        Returns:
            A new model with parameters taking values in the state_dict()

        """
        d_in = state_dict['w0'].shape[0]
        d_out = state_dict['w1'].shape[0]
        d_latent = state_dict['w0'].shape[1]

        mdl = cls(d_in, d_out, d_latent)
        mdl.load_state_dict(state_dict)
        return mdl

    def init_weights(self, y: torch.Tensor):
        """ Randomly initializes all model parameters based on data.

        This function should be called before model fitting.

        Args:
            y: Output data that the model will be fit to of shape n_smps*d_out

        Raise:
            NotImplementedError: If a parameter of the model exists for which initialization code
            does not exist.
        """

        d_in = self.w0.shape[0]
        d_out = self.w1.shape[0]

        var_variance = np.reshape(np.var(y.numpy(), 0), [d_out, 1])

        for param_name, param in self.named_parameters():
            if param_name in {'g'}:
                param.data = 10*torch.ones_like(param.data)
            elif param_name in {'v'}:
                param.data = torch.ones_like(param.data)
            elif param_name in {'o2'}:
                param.data = torch.zeros_like(param.data)
            elif param_name in {'w0'}:
                param.data.normal_(0, 1/np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, 1)
            else:
                raise(NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def generate_random_model(self, var_range: list = [.5, 1], g_range: list = [5, 10],
                              o2_range: list = [0, 1], w_offsets: list = [0, -1], w_gains: list = [1, 1]):
        """ Generates random values for model parameters.

        This function is useful for when generating models for testing code.

        Args:
            var_range: A list giving limits of a uniform distribution variance values will be pulled from

            g_range: A list giving limits of a uniform distribution g values will be pulled from

            o2_range: A list giving limits of a uniform distribution o2 values will be pulled from

            w_offsets: Means of the normal distributions weights are pulled from w_offsets[i] is the mean for w_i.
            Setting w_offsets[1] to be negative, helps ensure (assuing input has enough variance) that the generated
            data uses the full shape of the exponential - which can be important for practical concerns about model
            identifiability.

            w_gains: Gains of the standard deviations of the normal distributions weights are pulled from. Weights
            for w_0 will be pulled from a normal distribution with standard deviation w_gains[0]/sqrt(d_in) and weights
            for w_1 will be pulled from a normal distribution with standard deviation w_gains[1].

        """

        d_in = self.w0.shape[0]

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data.uniform_(*var_range)
            elif param_name in {'g'}:
                param.data.uniform_(*g_range)
            elif param_name in {'o2'}:
                param.data.uniform_(*o2_range)
            elif param_name in {'w0'}:
                param.data.normal_(w_offsets[0], w_gains[0] / np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(w_offsets[1], w_gains[1])
            else:
                raise (NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output means condition on x.

        This is equivalent to running the full generative model but not adding noise from n_t.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            mns: The output.  Of shape n_smps*d_out

        """
        x = torch.matmul(torch.t(self.w0), torch.t(x))
        x = torch.matmul(self.w1, x)
        x = torch.exp(x)
        x = self.g*x
        x = x + self.o2
        return torch.t(x)

    def latent_rep(self, x:torch.Tensor) -> List[torch.Tensor]:
        """ Computes unobserved intermediate values in the model.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            mn: The value of g_out + o_2 (see below)

            g_out: The value of g*exp(exp_in) (see below)

            exp_in: The value of w1*l (see l below)

            l: The value of w_0^T*x_t
        """
        with torch.no_grad():
            l = torch.matmul(torch.t(self.w0), torch.t(x))
            exp_in = torch.matmul(self.w1, l)
            exp_out = torch.exp(exp_in)
            g_out = self.g * exp_out
            mn = g_out + self.o2
            return [torch.t(mn), torch.t(g_out), torch.t(exp_in), torch.t(l)]


class RRReluModel(RRLinearModel):
    """ Rectified linear reduced rank model.

    For models of the form:

        y_t = relu(w_1*w_0^T * x_t + o_1) + o_2 + n_t,

        n_t ~ N(0, V), for a diagonal V

    where x_t is the input and relu() is the element-wise application of the relu.  We assume w_0 and w_1 are
    tall and skinny matrices.

    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        """ Create a RRReluModel object.

        Args:
            d_in: Input dimensionality

            d_out: Output dimensionality

            d_latent: Latent dimensionality
        """
        super().__init__(d_in, d_out, d_latent)

        o1 = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('o1', o1)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        """ Creates a new RRReluModel model from a dictionary.

        Args:
            state_dict: The state dictionary to create the model from.  This can be obtained
            by calling .state_dict() on a model.

        Returns:
            A new model with parameters taking values in the state_dict()

        """
        d_in = state_dict['w0'].shape[0]
        d_out = state_dict['w1'].shape[0]
        d_latent = state_dict['w0'].shape[1]

        mdl = cls(d_in, d_out, d_latent)
        mdl.load_state_dict(state_dict)
        return mdl

    def init_weights(self, y: torch.Tensor, w_gain: float = .5):
        """ Randomly initializes all model parameters based on data.

        This function should be called before model fitting.

        Args:
            y: Output data that the model will be fit to of shape n_smps*d_out

            w_gain: Weights will be initialized from a N(0, w_gain/sqrt(d_in)) distribution
            for w_0 and N(0, w_gain) distribution from w1.

        Raise:
            NotImplementedError: If a parameter of the model exists for which initialization code
            does not exist.
        """

        d_in = self.w0.shape[0]
        d_out = self.w1.shape[0]

        var_variance = np.reshape(np.var(y.numpy(), 0), [d_out, 1])
        var_min = np.reshape(np.min(y.numpy(), 0), [d_out,1])

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data = torch.from_numpy(var_variance/2)
            elif param_name in {'o2'}:
                param.data = torch.from_numpy(var_min)
            elif param_name in {'o1'}:
                param.data = torch.zeros_like(param.data)
            elif param_name in {'w0'}:
                param.data.normal_(0, w_gain/np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, w_gain)
            else:
                raise(NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def generate_random_model(self, var_range: list = [.5, 1], o1_range: list = [-.2, .2],
                              o2_range: list = [5, 10], w_gain: float = 1.0):
        """ Generates random values for model parameters.

        This function is useful for when generating models for testing code.

        Args:
            var_range: A list giving limits of a uniform distribution variance values will be pulled from

            o1_range: A list giving limits of a uniform distribution o1 values will be pulled from

            o2_range: A list giving limits of a uniform distribution o2 values will be pulled from

            w_gain: Entries of w0 and w1 are pulled from a distribution with a standard deviation of w_gain/sqrt(d_in),
            and entries of w1 are pulled from a distribution with a standard deviation of w_gain

        """

        d_in = self.w0.shape[0]

        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data.uniform_(*var_range)
            elif param_name in {'o1'}:
                param.data.uniform_(*o1_range)
            elif param_name in {'o2'}:
                param.data.uniform_(*o2_range)
            elif param_name in {'w0'}:
                param.data.normal_(0, w_gain / np.sqrt(d_in))
            elif param_name in {'w1'}:
                param.data.normal_(0, w_gain)
            else:
                raise (NotImplementedError('Initialization for ' + param_name + ' is not implemented.'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output means condition on x.

        This is equivalent to running the full generative model but not adding noise from n_t.

        Args:
            x: Input of shape n_smps*d_in

        Returns:
            mns: The output.  Of shape n_smps*d_out

        """
        x = torch.matmul(torch.t(self.w0), torch.t(x))
        x = torch.matmul(self.w1, x)
        x = x + self.o1
        x = torch.relu(x)
        x = x + self.o2
        return torch.t(x)

    def vary_latents(self, p_mat: torch.Tensor, mn_input: torch.Tensor, latent_vls: torch.Tensor) -> list:
        """ Produces output of model as latents take on different values.

        We define the latents for the model as l_t = w_0^T*x_t, for l_t \in R^d_latent.  We want to know
        what happens to the model as we change values in different dimensions of our latent space.

        Generally, we want to be able to work with arbitrary coordinate systems in our latent space, so we introduce
        projection matrices to allow us to pick out latent dimensions we want to vary. Let proj_m be a projection matrix
        such that proj_m = p_mat*p_mat^T, and let proj_c = I - proj_m.  In this was, the columns of p_mat specify latent
        dimensions in R^d_latent we are going to explore.

        For latent_vls[:,i] this function will compute:

            delta[:, t] = relu(w_1*p_mat*latent_vls[:,i] + mean_c + o_1) + o_2, where mean_c = w_1*proj_c*w_0^T*mn_input.

        Intuitively, we are doing the following.  We are saying (1), let's assume that all of the latent dimensions
        we are not varying are at their mean value.  Now (2), what happens to the output of my function as I vary the
        latent dimensions specified by p_mat.  To make this clear, notice that p_mat*latent_vls[:,i] gives the
        value of the latent dimensions we are interested in.

        This function will also return w_1_proj = w_1*p_mat and w_0_proj = w_0*p_mat, so that activation of the w1 and
        w0 weights corresponding to latent_vls[:,i] can be computed as w_1_proj*latent_vls[:,i] and
        w_0_proj*latent_vls[:,i].

        Note: Gradients will not be updated during this function call.

        Args:
            p_mat: A matrix with orthonormal columns defining the latent space to project into.

            mn_input: The mean input vector to calculate changes of input around.  Note, as specified above, any
            component of proj_m*w_0^T*x_t will be ignored.  Thus, we can interpret this function as giving changes
            from the mean activation if the components corresponding to p_mat are 0.

            latent_vls: An array of latent values to generate output for of shape n_smps*d_vary, where d_vary is the
            number of latent dimensions varied.

        Returns:
            delta: The model output for each point in latent_vls as a tensor.  Of shape n_smps*d_out

            w_0_proj: A tensor equal o w_0*p_mat of shape d_in*d_latent

            w_1_proj: A tensor equal to w_1*p_mat of shape d_out*d_latent

        Raises:
            ValueError: If the columns of p_mat are not orthonormal.
        """

        n_proj_dims = p_mat.shape[1]
        d_latent = self.w0.shape[1]

        if len(mn_input.shape) == 1:
            mn_input = torch.unsqueeze(mn_input, 1)

        with torch.no_grad():
            # First, make sure that p_mat is a matrix with orthonormal columns
            check = torch.matmul(torch.t(p_mat), p_mat)
            check_diff = check - torch.eye(n_proj_dims, dtype=check.dtype)
            if not np.all(np.isclose(check_diff.detach().numpy(), np.zeros(check_diff.shape))):
                raise(ValueError('p_mat must have orthonormal columns.'))

            # Now compute projected weights
            w_0_proj = torch.matmul(self.w0, p_mat)
            w_1_proj = torch.matmul(self.w1, p_mat)

            proj_c = torch.eye(d_latent) - torch.matmul(p_mat, torch.t(p_mat))
            mean_c = torch.matmul(self.w1, torch.matmul(proj_c, torch.matmul(torch.t(self.w0), mn_input)))

            # Now compute delta
            x = torch.matmul(w_1_proj, latent_vls)
            x = x + mean_c + self.o1
            x = torch.relu(x)
            x = x + self.o2
            delta = torch.t(x)

            return [delta, w_0_proj, w_1_proj]




