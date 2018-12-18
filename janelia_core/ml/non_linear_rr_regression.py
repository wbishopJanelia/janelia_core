""" Contains classes and functions for performing different forms of nonlinear reduced
rank regression.

    William Bishop
    bishopw@hhmi.org
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from janelia_core.visualization.matrix_visualization import cmp_n_mats


class NonLinearRRRegresion(torch.nn.Module):
    """ Basic sigmoidal non-linear reduced rank regression.

    Fits models of the form:

        y_t = sig(w_1*w_0^T * x_t + o_1) + o_2 + n_t,

        n_t ~ N(0, V), for a diagonal V

    where x_t is the input and sig() is the element-wise application of the sigmoid.  We assume w_0 and w_1 are
    tall and skinny matrices.

    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        """ Create a NonLinearRRRegrssion object.

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

        o1 = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('o1', o1)

        g = torch.nn.Parameter(torch.zeros([d_out]), requires_grad=True)
        self.register_parameter('g', g)

        o2 = torch.nn.Parameter(torch.zeros([d_out, 1]), requires_grad=True)
        self.register_parameter('o2', o2)

        # v is the *variances* of each noise term
        v = torch.nn.Parameter(torch.zeros([d_out]), requires_grad=True)
        self.register_parameter('v', v)

    def init_weights(self):
        """ Randomly initializes all model parameters."""
        for param_name, param in self.named_parameters():
            if param_name in {'v'}:
                param.data.uniform_(0, 1)
            elif param_name in {'g'}:
                param.data.uniform_(0, 10)
            else:
                param.data.normal_(0, 1)

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
        x = torch.matmul(torch.diag(self.g), x)
        x = x + self.o2
        return torch.t(x)

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
            noise = torch.randn_like(mns)*torch.sqrt(self.v)
            return mns + noise

    def neg_log_likelihood(self, y: torch.Tensor, mns: torch.Tensor):
        """ Calculates the negative log-likelihood of observed data, given conditional means for that
        data up to a constant.

        Note: This function does not compute the term .5*n_smps*log(2*pi).  Add this term in
        if you want the exact log likelihood.  Also, this function divides the negative log likelihood
        by the number of samples.

        This function can be used as a loss, using the output of forward for mns and setting
        y to be observed data.  Using this function as loss will give (subject to local optima)
        MLE solutions.

        Args:
            y: Data to measure the negative log likelihood for of shape n_smps*d_in.

            mns: The conditional means for the data.  These can be obtained with forward().

        Returns:
            nll: The negative log-likelihood of the observed data.
        """

        n_smps = y.shape[0]
        return .5*torch.sum(torch.log(self.v)) + .5*torch.sum(((y - mns)**2)/self.v)/n_smps

    def fit(self, x: torch.Tensor, y: torch.Tensor, batch_size: int=100, max_its: int=10,
            adam_params: dict = {'lr': .01}):
        """ Fits a model to data.

        This function performs stochastic optimization with the ADAM algorithm.  The weights of the model
        should be initialized before calling this function.

        Optimization will be perfomed on whatever device the model parameters are on.

        Args:

            x: Tensor of input data of shape n_smps*d_in

            y: Tensor of output data of shape n_smps*d_out

            batch_size: The number of samples to train on during each iteration

            max_its: The maximum number of iterations to run

            adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object

            """

        device = self.g.device

        optimizer = torch.optim.Adam(self.parameters(), **adam_params)

        n_smps = x.shape[0]
        cur_it = 0
        start_time = time.time()

        elapsed_time_log = np.zeros(max_its)
        nll_log = np.zeros(max_its)

        while cur_it < max_its:

            # Chose the samples for this iteration
            cur_smps = np.random.choice(n_smps, batch_size, replace=False)
            send_start = time.time()
            batch_x = x[cur_smps, :].to(device)
            batch_y = y[cur_smps, :].to(device)
            send_end = time.time()
            print('Data send time: ' + str(send_end - send_start))

            # Perform optimization for this step
            opt_start = time.time()
            optimizer.zero_grad()
            mns = self(batch_x.data)
            nll = self.neg_log_likelihood(batch_y.data, mns)
            nll.backward()
            optimizer.step()
            opt_end = time.time()
            print('Optimization time: ' + str(opt_end - opt_start))

            # Log our progress
            log_start = time.time()
            elapsed_time = time.time() - start_time
            elapsed_time_log[cur_it] = elapsed_time
            cur_nll = nll.cpu().data.detach().numpy()
            nll_log[cur_it] = cur_nll
            log_end = time.time()
            print('Logging time: ' + str(log_end - log_start))

            # Provide user with some feedback
            if cur_it % 1 == 0:
                print(str(cur_it) + ': Elapsed time ' + str(elapsed_time) +
                      ', vl: ' + str(cur_nll))

            cur_it += 1

    def standardize(self):
        """ Puts the model in a standard form.

        The models have multiple degenerecies (non-identifiabilities):

            1) The signs of gains (g) is arbitrary.  A change in sign can be absorbed by a change
            in w1, o1 and d.  This function will put models in a form where all gains have positive
            sign.

            2) Even after (1), the values of w1 and w2 are not fully determined.  After standardizing
            with respect to gains, the svd of w1 = u1*s1*v1.T will be performed (where s1 is a non-negative
            diagonal matrix) and then w1 will be set w1=u1 and w0 will be set w0 = wo*s1*v1
        """

        # Standardize with respect to gains
        for i in range(len(self.g)):
            if self.g[i] < 0:
                print('Changing sign of gain ' + str(i))
                self.w1.data[i, :] = -1*self.w1.data[i, :]
                self.o1.data[i] = -1*self.o1.data[i]
                self.o2.data[i] = self.o2.data[i] + self.g.data[i]
                self.g.data[i] = -1*self.g.data[i]

        # Standardize with respect to weights
        [u1, s1, v1] = torch.svd(self.w1.data, some=True)
        self.w1.data = u1
        self.w0.data = torch.matmul(torch.matmul(self.w0.data, v1), torch.diag(s1))

    @staticmethod
    def compare_models(m1, m2, x: torch.Tensor = None, plot_vars: int = 2):
        """ Visually compares two models.

        Args:
            m1: The fist model

            m2: The second model

            x: Input to test the models on.  The conditional means for both model will be plotted for this input if
            provided.

            plot_vars: Indices of variables to plot if plotting conditional means. If this None, the first (up to) two
            variables will be plotted.

        """

        ROW_SPAN = 14  # Number of rows in the gridspec
        COL_SPAN = 12 # Number of columns in the gridspec

        grid_spec = matplotlib.gridspec.GridSpec(ROW_SPAN, COL_SPAN)

        def _make_subplot(loc, r_span, c_span, d1, d2, title):
            subplot = plt.subplot(grid_spec.new_subplotspec(loc, r_span, c_span))
            subplot.plot(d1, 'b-')
            subplot.plot(d2, 'r-')
            subplot.title = plt.title(title)

        def _make_weight_subplots(loc, r_span, c_span, w, title):
            subplot = plt.subplot(grid_spec.new_subplotspec(loc, r_span, c_span))
            subplot.imshow(w)
            subplot.title = plt.title(title)

        # Make plots of scalar variables
        _make_subplot([0, 0], 2, 2, m1.g.detach().numpy(), m2.g.detach().numpy(), 'g')
        _make_subplot([3,0], 2, 2, m1.o2.detach().numpy(), m2.o2.detach().numpy(), 'o2')
        _make_subplot([6, 0], 2, 2, m1.o1.detach().numpy(), m2.o1.detach().numpy(), 'o1')
        _make_subplot([9, 0], 2, 2, m1.v.detach().numpy(), m2.v.detach().numpy(), 'v')

        # Make plots of w1 matrices
        m1_w1 = m1.w1.detach().numpy()
        m2_w1 = m2.w1.detach().numpy()
        w1_diff = m1_w1 - m2_w1
        w1_grid_info = {'grid_spec': grid_spec}
        w1_cell_info = list()
        w1_cell_info.append({'loc': [0, 2], 'rowspan': 10, 'colspan': 1})
        w1_cell_info.append({'loc': [0, 4], 'rowspan': 10, 'colspan': 1})
        w1_cell_info.append({'loc': [0, 6], 'rowspan': 10, 'colspan': 1})
        w1_grid_info['cell_info'] = w1_cell_info
        cmp_n_mats([m1_w1, m2_w1, w1_diff], show_colorbars=True, titles=['m1_w1', 'm2_w1', 'm1_w1 - m2_w1'], grid_info=w1_grid_info)

        # Make plots of w0 matrices
        m1_w0 = m1.w0.detach().numpy().T
        m2_w0 = m2.w0.detach().numpy().T
        w0_diff = m1_w0 - m2_w0
        w0_grid_info = {'grid_spec': grid_spec}
        w0_cell_info = list()
        w0_cell_info.append({'loc': [0, 8], 'rowspan': 1, 'colspan': COL_SPAN-8+1})
        w0_cell_info.append({'loc': [2, 8], 'rowspan': 1, 'colspan': COL_SPAN-8+1})
        w0_cell_info.append({'loc': [4, 8], 'rowspan': 1, 'colspan': COL_SPAN-8+1})
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



