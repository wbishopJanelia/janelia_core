""" Torch modules for working with distributions.

Note: These are *not* subclasses of the torch.distributions.

 """

from typing import Sequence
import torch
import math


class CondVAEDistriubtion(torch.nn.Module):
    """ CondVAEDistribution is an abstract base class for distributions used by VAEs."""

    def __init__(self):
        """ Creates a CondVAEDistribution object. """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes the conditional mean of the distribtion at different samples.

        Args:
            x: A tensor of shape n_smps*d_x.

        Returns:
            mn: mn[i, :] is the mean conditioned on x[i, :]
        """
        raise NotImplementedError

    def sample(self, x: torch.Tensor) -> object:
        """ Samples from a conditional distribution.

        When possible, samples should be generated from a reparameterized distribution.

        Returned samples may be represented by a set of compact parameters.  See form_sample() on how to transform this
        compact represntation into a standard representation.

        Args:
            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

        Returns:
            smp: The sample. The returned value of samples can be quite flexible.  It could be a tensor of shape n_smps,
            with each entry representing a sample or it could be another object with attributes which specify the values
            of the sample.  For example, if sampling from a spike and slab parameter, the returned value could be a list
            with one entry specifying the number of sample, another containing a tensor specifying non-zero samples and
            another tensor specifying the values of the non-zero samples.
        """
        raise NotImplementedError

    def form_standard_sample(self, smp: object) -> torch.Tensor:
        """ Forms a standard representation of a sample from the output of sample.

        Args:
            smp: Compact representation of a sample.

        Returns:
            formed_smp: A tensor of shape n_smps*d_y.  formed_smp[i,:] is the i^th sample.
        """
        raise NotImplementedError

    def form_compact_sample(self, smp: torch.Tensor) -> object:
        """ Forms a compact representation of a sample given a standard representation.

        Args:
            smp: The standard representation of the sample of shape n_smps

        Returns:
            formed_smp: The compact representation of the sample.
        """
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, y: object) -> torch.Tensor:
        """ Computes the conditional log probability of individual samples.

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Compact representation of the samples we desire the probability for.  Compact representation means the
            form of a sample as output by the sample() function.

        Returns:
            ll: Conditional log probability of each sample. Of shape n_smps.
        """
        raise NotImplementedError


class CondBernoulliDistribution(CondVAEDistriubtion):
    """ A module for working with conditional Bernoulli distributions."""

    def __init__(self, log_prob_fcn: torch.nn.Module):
        """ Creates a BernoulliCondDistribution variable.

        Args:
            log_prob_fcn: A function which accepts input of shape n_smps*d_x and outputs a tensor of shape n_smps with
            the log probability that each sample is 1.
        """

        super().__init__()
        self.log_prob_fcn = log_prob_fcn

    def forward(self, x: torch.Tensor):
        """ Computes conditional mean given samples.

        Args:
            x: data samples are conditioned on. Of shape n_smps*d_x.

        Returns:
            mn: mn[i,:] is the mean conditioned on x[i,:]
        """

        nz_prob = torch.exp(self.log_prob_fcn(x))
        return nz_prob

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape nSmps*d_x.

            y: Compact representation (a tensor of type byte) of the sample.

        Returns:
            log_prob: the log probability of each sample

        Raises:
            ValueError: If y is not a 1-d tensor.

        """

        if len(y.shape) != 1:
            raise (ValueError('y must be a 1 dimensional tensor.'))

        n_smps = x.shape[0]

        zero_inds = y == 0

        log_nz_prob = self.log_prob_fcn(x)

        log_prob = torch.zeros(n_smps)
        log_prob[~zero_inds] = log_nz_prob[~zero_inds]
        log_prob[zero_inds] = torch.log(1 - torch.exp(log_nz_prob[zero_inds]))

        return log_prob

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """ Samples from P(y|x)

        Args:
            x: Data we condition on.  Of shape nSmps*d_x.

        Returns:
            smp: smp[i] is the value of the i^th sample.
        """

        probs = torch.exp(self.log_prob_fcn(x))
        bern_dist = torch.distributions.bernoulli.Bernoulli(probs)
        return bern_dist.sample().byte()

    def form_standard_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return  smp


class CondGaussianDistribution(CondVAEDistriubtion):
    """ Represents a multivariate distribution over a set of conditionally independent Gaussian random variables.
    """

    def __init__(self, mn_f: torch.nn.Module, std_f: torch.nn.Module):
        """ Creates a CondGaussianDistribution object.

        Args:
            mn_f: A module whose forward function accepts input of size n_smps*d_x and outputs a mean for each sample in a
                  tensor of size n_smps*d_y

            std_f: A module whose forward function accepts input of sixe n_smps*d and outputs a standard deviation for
                   each sample of size n_smps*d_y

        """

        super().__init__()

        self.mn_f = mn_f
        self.std_f = std_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes conditional mean given samples.

        Args:
            x: data samples are conditioned on. Of shape n_smps*d_x.

        Returns:
            mn: mn[i,:] is the mean conditioned on x[i,:]
        """

        return self.mn_f(x)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Values we desire the log probability for.  Of shape nSmps*d_y.

        Returns:
            ll: Log-likelihood of each sample. Of shape n_smps.

        """

        d_x = x.shape[1]

        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        mn = self.mn_f(x)
        std = self.std_f(x)

        ll = -.5*torch.sum(((y - mn)/std)**2, 1)
        ll -= torch.sum(torch.log(std), 1)
        ll -= .5*d_x*torch.log(torch.tensor([math.pi]))

        return ll

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """ Samples from the reparameterized form of P(y|x).

        If a sample without gradients is desired, wrap the call to sample in torch.no_grad().

        Args:
            x: Data we condition on.  Of shape nSmps*d_x.

        Returns:
            y: sampled data of shape nSmps*d_y.
        """

        mn = self.mn_f(x)
        std = self.std_f(x)

        n_smps = mn.shape[0]
        d_x = mn.shape[1]

        z = torch.randn(n_smps, d_x)

        return mn + z*std

    def form_standard_sample(self, smp):
        return smp

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp


class CondLowRankMatrixDistribution(torch.nn.Module):

    """ Represents a conditional product distribution over the left and right components of a low-rank matrix.
    """

    def __init__(self, l_mode_dists: Sequence[CondVAEDistriubtion], r_mode_dists: Sequence[CondVAEDistriubtion]):
        """ Creates a CondLowRankMatrixDistribution object.

        This distribution represents a conditional distribution over matrices of the form M = LR^T for L \in R^n*p
        and R^m*p.

        Rows or L and R are conditioned on properties contained in the matrices X_L \in R^n*d_l and
        X_R \in R^m*d_r.

        The probability of a pair L, R can then be written as:

            P(L, R | X_L, X_R) =  (\prod_i=1^n P(L[i,:] | X_L[i,:])) * (\prod_j=1^m P(R[j,:] | X_R[j,:])).

        Likewise P(L[i,:] | X_L[i,:]) can be written as:

            P(L[i,:] | X_L[i,:]) = \prod_k=1^k P(L[i,k] | X_L[i,:]).

        And P(R[j,:] | X_R[j,:]) can be written in a similar manner.

        Args:
            l_mode_dists: l_mode_dists[i] gives the conditional distribution specifying P(L[i,:] | X_L[i,:]).

            r_mode_dists: r_mode_dists[j] gives the conditional distribution specifying P(R[j,:] | X_R[j,:])
        """

        super().__init__()

        self.n_modes = len(l_mode_dists)

        self.l_mode_dists = l_mode_dists
        self.r_mode_dists = r_mode_dists

    def sample(self, x_l: torch.Tensor, x_r: torch.Tensor) -> list:
        """ Returns a compact representation of samples for each mode conditioned on x_l and x_r.

        Args:
            x_l: X_L tensor to condition on.

            x_r: X_R tensor to condition on.

        Returns:
            s_l: s_l[m] contains a compact representation of the sample for the entries of L[:,m].

            s_r: s_r[m] contains a compact representation of the sample for the entries of R[:,m].
        """

        s_l = [self.l_mode_dists[m].sample(x_l) for m in range(self.n_modes)]
        s_r = [self.r_mode_dists[m].sample(x_r) for m in range(self.n_modes)]

        return [s_l, s_r]


    def log_prob(self, x_l: torch.Tensor, x_r: torch.Tensor, s_l: list, s_r: list) -> torch.Tensor:
        """ Computes the log conditional probability of L and R matrices given X_L and X_R.

        Args:
            x_l: X_L tensor to condition on.

            x_r: X_R tensor to condition on.

            s_l: Compact representation (as returned by sample) of entries of L

            s_r: Compact representation (as returned by sample) of entries of R

        Returns:
            ll: The conditional log-probability of L and R as a scalar tensor.
        """

        # Compute log probability for each mode (summing over entries)
        l_log_probs = [torch.sum(self.l_mode_dists[m].log_prob(x_l, s_l[m])) for m in range(self.n_modes)]
        r_log_probs = [torch.sum(self.r_mode_dists[m].log_prob(x_r, s_r[m])) for m in range(self.n_modes)]

        # Compute log probability for all modes (summing over modes)
        return torch.sum(l_log_probs) + torch.sum(r_log_probs)


class CondSpikeSlabDistribution(CondVAEDistriubtion):
    """ Represents a condition spike and slab distriubtion. """

    def __init__(self, d: int, spike_d: CondVAEDistriubtion, slab_d: CondVAEDistriubtion):
        """ Creates a CondSpikeSlabDistribution object.

        Args:

            d: The number of variables the spike and slab distribution is over

            spike_d: The spike distribution

            slab_d: The slab distribution
        """
        super().__init__()

        self.d = d
        self.spike_d = spike_d
        self.slab_d = slab_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Values we desire the log probability for.  Of shape nSmps*d_y.

        Returns:
            ll: Log-likelihood of each sample. Of shape n_smps.

        """
        n_smps = x.shape[0]

        spike_p = torch.exp(self.spike_d.log_prob(x, torch.ones(n_smps))).unsqueeze(1)
        slab_mn = self.slab_d(x)

        return spike_p*slab_mn

    def sample(self, x: torch.Tensor) -> list:
        """ Samples a conditional spike and slab distribution.

        This function will return samples in compact form.

        Args:
            x: The data to condition on.  Of shape n_smps*d_x.

        Returns: A compact representation of the sample:

            n_smps: the number of samples

            support: A binary tensor. support[i] is 1 if smp i is non-zero

            nz_vls: A tensor with the non-zero values.  nz_vls[j,:] contains the value for the j^th non-zero entry in
                    support. In other words, nz_vls gives the non-zero values corresponding to the samples in
                    x[support, :].  If there are no non-zero values this will be None.

        """

        n_smps = x.shape[0]
        support = self.spike_d.form_standard_sample(self.spike_d.sample(x))
        if any(support):
            nz_vls = self.slab_d.sample(x[support, :]).squeeze()
        else:
            nz_vls = None

        return [n_smps, support, nz_vls]

    def log_prob(self, x: torch.Tensor, y: list) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x.

            y: Compact representation of a sample.  See sample().

        Returns:
            ll: Log-likelihood of each sample.

        """

        n_smps, support, nz_vls = y

        # Log-likelihood due to spike distribution
        ll = self.spike_d.log_prob(x, support)
        # Log-likelihood due to slab distribution
        if any(support):
            ll[support] += self.slab_d.log_prob(x[support, :], nz_vls)

        return ll

    def form_standard_sample(self, smp) -> torch.Tensor:
        """ Forms a standard sample representation from a compact representation.

        Args:
           smp: The compact representation of a sample (the compact representation of a sample is the form returned by
           sample)

        Returns:
             formed_smp: The standard form of a sample.  formed_smp[i] gives the value of the i^th sample.
        """

        n_smps, support, nz_vls = smp

        # First handle the case where all values are zero
        if nz_vls is None:
            return torch.zeros([n_smps, self.d])

        # Now handle the case where we have at least one non-zero value
        if len(nz_vls.shape) > 1:
            formed_smp = torch.zeros([n_smps, self.d])
            formed_smp[support, :] = nz_vls
        else:
            formed_smp = torch.zeros(n_smps)
            formed_smp[support] = nz_vls

        return formed_smp

    def form_compact_sample(self, smp: torch.Tensor) -> list:
        """ Forms a compact sample from a full sample.

        Args:
            smp: The standard representation of the sample of shape n_smps*d

        Returns:
            n_smps, support, nz_vls: Compact representation of the sample.  See sample().
        """

        n_smps = smp.shape[0]

        if self.d > 1:
            support = torch.sum(smp != 0, dim=1) == self.d
        else:
            support = smp != 0

        if any(support):
            nz_vls = smp[support,:]
        else:
            nz_vls = None

        return [n_smps, support, nz_vls]
