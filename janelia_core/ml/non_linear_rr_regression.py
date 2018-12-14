""" Contains classes and functions for performing different forms of nonlinear reduced
rank regression.

    William Bishop
    bishopw@hhmi.org
"""

import torch


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

    def neg_log_likelihood(self, y, mns):
        """ Calculates the negative log-likelihood of observed data, given conditional means for that
        data up to a constant.

        Note: This function does not compute the term .5*n_smps*log(2*pi).  Add this term in
        if you want the exact log likelihood

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
        return .5*n_smps*torch.sum(torch.log(self.v)) + .5*torch.sum(((y - mns)**2)/self.v)

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
                self.w1.data[i, :] = -1*self.w1.data[i, :]
                self.o1.data[i] = -1*self.o1.data[i]
                self.o2.data[i] = self.o2.data[i] + self.g.data[i]
                self.g.data[i] = -1*self.g.data[i]

        # Standardize with respect to weights
        [u1, s1, v1] = torch.svd(self.w1.data, some=True)
        self.w1.data = u1
        self.w0.data = torch.matmul(torch.matmul(self.w0.data, v1), torch.diag(s1))
