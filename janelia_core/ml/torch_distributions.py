""" Torch modules for working with distributions.

Note: These are *not* subclasses of the torch.distributions.

 """

import itertools
from typing import Sequence

import torch
import math


class CondVAEDistriubtion(torch.nn.Module):
    """ CondVAEDistribution is an abstract base class for distributions used by VAEs."""

    def __init__(self):
        """ Creates a CondVAEDistribution object. """
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes the conditional mean of the distribtion at different samples.

        Args:
            x: A tensor of shape n_smps*d_x.

        Returns:
            mn: mn[i, :] is the mean conditioned on x[i, :]
        """
        raise NotImplementedError

    def sample(self, x: torch.tensor) -> object:
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

    def form_standard_sample(self, smp: object) -> torch.tensor:
        """ Forms a standard representation of a sample from the output of sample.

        Args:
            smp: Compact representation of a sample.

        Returns:
            formed_smp: A tensor of shape n_smps*d_y.  formed_smp[i,:] is the i^th sample.
        """
        raise NotImplementedError

    def form_compact_sample(self, smp: torch.tensor) -> object:
        """ Forms a compact representation of a sample given a standard representation.

        Args:
            smp: The standard representation of the sample of shape n_smps

        Returns:
            formed_smp: The compact representation of the sample.
        """
        raise NotImplementedError

    def log_prob(self, x: torch.tensor, y: object) -> torch.tensor:
        """ Computes the conditional log probability of individual samples.

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Compact representation of the samples we desire the probability for.  Compact representation means the
            form of a sample as output by the sample() function.

        Returns:
            ll: Conditional log probability of each sample. Of shape n_smps.
        """
        raise NotImplementedError

    def kl(self, d_2, x: torch.tensor, smp: Sequence = None):
        """ Computes the KL divergence between this object and another of the same type conditioned on input.

        Specifically computes:

            KL(p_1(y_i|x_i), p_2(y_i|x_i)),

        where p_1(y_i | x_i) represents the conditional distributions for each sample.  Here, p_1 is the conditional
        distribution represented by this object and p_2 is the distribution represented by another object of the same
        type.

        Args:
            d_2: The other conditional distribution in the KL divergence.  Must be of the same type as this object.  If
            a multivariate distribution, must also be over random variables as the same size as the random variables
            the distribution for this object is over.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: An set samples of shape n_smps*d_y. smp[i,:] should be drawn from p(y_i|x[i,:]). This is an optional
            input that is provided because sometimes it may not be possible to compute the KL divergence
            between two distributions analytically.  In these cases, an object may still implement the kl method
            by computing an empirical estimate of the kl divergence as log p_1(y_i'|x_i) - log p_2(y_i'| x_i),
            where y_i' is drawn from p_1(y_i|x_i). This is the base behavior of this method.  Objects for which kl
            can be computed analytically should override this method.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th sample.

        Raises:
            ValueError: if d2 is not the same type as this object
        """
        if type(d_2) != type(self):
            raise(ValueError('KL divergence must be computed between distributions of the same type.'))

        kl = self.log_prob(x, smp) - d_2.log_prob(x, smp)
        return kl.squeeze()

    def r_params(self) -> list:
        """ Returns a list of parameters for which gradients can be estimated with the reparameterization trick.

        In particular this returns the list of parameters for which gradients can be estimated with the
        reparaterization trick when the distribution serves as q when optimizing KL(q, p).

        If no parameters can be estiamted in this way, should return an empty list.

        Returns:
            l: the list of parameters

        """
        raise NotImplementedError

    def s_params(self) ->list:
        """ Returns a list of parameters which can be estimated with a score method based gradient.

        In particular this returns the list of parameters for which gradients can be estimated with the
        score function based gradient when the distribution serves as q when optimizing KL(q, p).

        If no parameters can be estiamted in this way, should return an empty list.

        Returns:
            l: the list of parameters
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

    def r_params(self) -> list:
        return list()

    def s_params(self) -> list:
        return list(self.parameters())


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

        self.register_buffer('log_2_pi', torch.log(torch.tensor(2*math.pi)))

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

            y: Values we desire the log probability for.  Of shape n_smps*d_y.

        Returns:
            ll: Log-likelihood of each sample. Of shape n_smps.

        """

        d_x = x.shape[1]

        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        d_y = y.shape[1]

        mn = self.mn_f(x)
        std = self.std_f(x)

        ll = -.5*torch.sum(((y - mn)/std)**2, 1)
        ll -= torch.sum(torch.log(std), 1)
        ll -= .5*d_y*self.log_2_pi

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

        z = torch.randn_like(std)

        return mn + z*std

    def kl(self, d_2, x: torch.tensor, smp: Sequence = None):
        if type(d_2) != type(self):
            raise(ValueError('KL divergence must be computed between distributions of the same type.'))

        mn_1 = self.mn_f(x)
        std_1 = self.std_f(x)

        mn_2 = d_2.mn_f(x)
        std_2 = d_2.std_f(x)

        d = mn_1.shape[1]

        log_det_1 = 2*torch.sum(torch.log(std_1), dim=1)
        log_det_2 = 2*torch.sum(torch.log(std_2), dim=1)
        log_det_diff = log_det_2 - log_det_1

        sigma_ratio_sum = torch.sum((std_1**2)/(std_2**2), dim=1)

        mn_diff = torch.sum(((mn_2 - mn_1)/std_2)**2, dim=1)

        kl = .5*(log_det_diff - d + sigma_ratio_sum + mn_diff)

        return kl.squeeze()

    def form_standard_sample(self, smp):
        return smp

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp

    def r_params(self):
        return list(self.parameters())

    def s_params(self) -> list:
        return list()


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

            P(L[i,:] | X_L[i,:]) = \prod_k=1^p P(L[i,k] | X_L[i,:]).

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

    def log_prob(self, x_l: torch.Tensor, x_r: torch.Tensor, s_l: list, s_r: list) -> list:
        """ Computes the log conditional probability of L and R matrices given X_L and X_R.

        Args:
            x_l: X_L tensor to condition on.

            x_r: X_R tensor to condition on.

            s_l: s_l[i] contains the compact representation (as returned by sample) of entries of L[:, i]

            s_r: s_r[i] contains the compact representation (as returned by sample) of entries of R[:, i]

        Returns:
            l_log_probs: l_log_probs[j,i] gives the conditional log probability for L[j,i]

            r_log_probs: r_log_probs[j,i] gives the conditional log probability for R[j,i]
        """

        # Compute log probability for each mode (summing over entries)
        l_log_probs = torch.stack([self.l_mode_dists[m].log_prob(x_l, s_l[m]) for m in range(self.n_modes)]).t()
        r_log_probs = torch.stack([self.r_mode_dists[m].log_prob(x_r, s_r[m]) for m in range(self.n_modes)]).t()

        # Compute log probability for all modes (summing over modes)
        return [l_log_probs, r_log_probs]

    def form_matrix(self, smp):
        """ Forms a matrix representation of a sample from a compact representation of the sample.

        Args:
             smp: The compact representation of the sample, as returned by sample()

        Returns:
            m: The matrix representation of the sample
        """

        l_modes = smp[0]
        r_modes = smp[1]

        l0 = self.l_mode_dists[0].form_standard_sample(l_modes[0])
        r0 = self.r_mode_dists[0].form_standard_sample(r_modes[0])
        n = l0.numel()
        m = r0.numel()

        mat = torch.zeros((n, m))

        n_modes = len(l_modes)
        for m_i in range(n_modes):
            if m_i > 0:
                l = self.l_mode_dists[m_i].form_standard_sample(l_modes[m_i])
                r = self.r_mode_dists[m_i].form_standard_sample(r_modes[m_i])
            else:
                # Save computation since we already form the standard sample for the first modes
                l = l0
                r = r0
            l = l.view([n, 1])
            r = r.view([m, 1])
            mat += torch.matmul(l, r.t())

        return mat

    def r_params(self) -> list:
        """ Mimicks the behavior of r_params for CondVAEDistribution objects. """

        r_mode_params = itertools.chain(*[d.r_params() for d in self.r_mode_dists])
        l_mode_params = itertools.chain(*[d.r_params() for d in self.l_mode_dists])

        return list(itertools.chain(r_mode_params, l_mode_params))

    def s_params(self) -> list:
        """ Mimicks the behavior of s_params for CondVAEDistribution objects. """

        r_mode_params = itertools.chain(*[d.s_params() for d in self.r_mode_dists])
        l_mode_params = itertools.chain(*[d.s_params() for d in self.l_mode_dists])

        return list(itertools.chain(r_mode_params, l_mode_params))


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
        """ Computes  E(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Values we desire the log probability for.  Of shape nSmps*d_y.

        Returns:
            mn: Conditional expectation. Of shape n_smps*d_y

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
            nz_vls = self.slab_d.sample(x[support, :])
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

    def r_params(self) -> list:
        return self.slab_d.r_params()

    def s_params(self) -> list:
        return self.spike_d.s_params()
