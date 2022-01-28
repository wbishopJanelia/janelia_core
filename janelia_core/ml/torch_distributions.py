""" Torch modules and tools for working with distributions.

Note: The distribution objects defined here are *not* subclasses of the torch.distributions.

 """

import copy
import math
from typing import Sequence

import numpy as np
import torch

from janelia_core.math.basic_functions import list_grid_pts
from janelia_core.ml.extra_torch_functions import log_cosh
from janelia_core.ml.extra_torch_modules import FixedOffsetExp
from janelia_core.ml.extra_torch_modules import IndSmpConstantBoundedFcn
from janelia_core.ml.extra_torch_modules import IndSmpConstantRealFcn
from janelia_core.ml.extra_torch_modules import SCC
from janelia_core.ml.extra_torch_modules import SumAlongDim
from janelia_core.ml.extra_torch_modules import SumOfTiledHyperCubeBasisFcns
from janelia_core.ml.extra_torch_modules import Tanh


class CondVAEDistribution(torch.nn.Module):
    """ CondVAEDistribution is an abstract base class for conditional distributions used by VAEs."""

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

    def sample_to(self, smp: object, device: torch.device) -> object:
        """ Moves a sample in compact form to a given device.

        This function is provided because different distributions may return samples in arbitrary objects,
        so a custom function may be needed to move a sample to a device.

        Args:
            smp: The sample to move.

            device: The device to move the sample to.

        """
        raise(NotImplementedError)

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

    def kl(self, d_2, x: torch.tensor, smp: object = None, return_device: torch.device = None):
        """ Computes the KL divergence between this object and another of the same type conditioned on input.

        Specifically computes:

            KL(p_1(y_i|x_i), p_2(y_i|x_i)),

        where p_1(y_i | x_i) represents the conditional distributions for each sample.  Here, p_1 is the conditional
        distribution represented by this object and p_2 is the distribution represented by another object of the same
        type.

        Note: This function will move the conditioning data (x) and the sample (smp) to the appropriate device(s)
        so calculations can be carried out without needing to move this object or the other conditional
        distribution between devices.

        Args:
            d_2: The other conditional distribution in the KL divergence.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: A set of samples in compact form. Sample i should be drawn from p(y_i|x[i,:]). This is an optional
            input that is provided because sometimes it may not be possible to compute the KL divergence
            between two distributions analytically.  In these cases, an object may still implement the kl method
            by computing an empirical estimate of the kl divergence as log p_1(y_i'|x_i) - log p_2(y_i'| x_i),
            where y_i' is drawn from p_1(y_i|x_i). This is the base behavior of this method.  Objects for which kl
            can be computed analytically should override this method.

            return_device: The device the calculated kl tensor should be returned to.  If None, this will
            be the device the first parameter of this object is on.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th conditioning
            input.

        """

        self_device = next(self.parameters()).device
        d_2_device = next(d_2.parameters()).device

        if return_device is None:
            return_device = self_device

        smp_self = self.sample_to(smp=smp, device=self_device)
        x_self = x.to(device=self_device)
        smp_d_2 = d_2.sample_to(smp=smp, device=d_2_device)
        x_d_2 = x.to(device=d_2_device)

        kl = self.log_prob(x=x_self, y=smp_self).to(return_device) - d_2.log_prob(x=x_d_2, y=smp_d_2).to(return_device)
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


class CondFoldedNormalDistribution(CondVAEDistribution):
    """ A multivariate conditional folded normal distribution.

    A folder normal distribution is the distribution on the random variable, Y = abs(Z), when Z is
    distributed N(\mu, \sigma^2). This object represents a conditional distribution over a set of random
    variables, each of which is independent and distributed according to a Folded Normal, conditioned on X.

    """

    def __init__(self, mu_f, sigma_f):
        """ Creates a new CondFoldedNormalDistribution object.

        Args:
            mu_f: A module whose forward function accepts input of size n_smps*d_x and outputs a vector of mu
            parameters for size n_smps*d_y

            sigma_f: A module whose forward function accepts input of sixe n_smps*d and outputs a vector of
            standard deviations for each sample of size n_smps*dy

        """

        super().__init__()

        self.mu_f = mu_f
        self.sigma_f = sigma_f

        self.register_buffer('log_constants', .5*torch.log(torch.tensor(2.0)) - .5*torch.log(torch.tensor(math.pi)))
        self.register_buffer('sqrt_2_over_pi', torch.sqrt(2/torch.tensor(math.pi)))

    def form_standard_sample(self, smp):
        return smp

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes conditional mean given samples.

        Args:
            x: data samples are conditioned on. Of shape n_smps*d_x.

        Returns:
            mn: mn[i,:] is the mean conditioned on x[i,:]
        """

        mu = self.mu_f(x)
        sigma = self.sigma_f(x)

        return (sigma*self.sqrt_2_over_pi*torch.exp(-(mu**2)/(2*(sigma**2))) +
                mu*torch.erf(mu/torch.sqrt(2*(sigma**2))))

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Values we desire the log probability for.  Of shape n_smps*d_y.

        Returns:
            ll: Log-likelihood of each sample. Of shape n_smps.

        """

        mu = self.mu_f(x)
        sigma = self.sigma_f(x)

        sigma_sq = sigma**2
        return torch.sum(self.log_constants - torch.log(sigma) - (y**2 + mu**2)/(2*sigma_sq) + log_cosh(mu*y/sigma_sq),
                         dim=1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """ Samples from the reparameterized form of P(y|x).

        If a sample without gradients is desired, wrap the call to sample in torch.no_grad().

        Args:
            x: Data we condition on.  Of shape nSmps*d_x.

        Returns:
            y: sampled data of shape nSmps*d_y.
        """

        mn = self.mu_f(x)
        std = self.sigma_f(x)

        z = torch.randn_like(std)

        return torch.abs(mn + z*std)

    def sample_to(self, smp: object, device: torch.device):
        return smp.to(device)

    def r_params(self):
        return list(self.parameters())

    def s_params(self) -> list:
        return list()


class CondBernoulliDistribution(CondVAEDistribution):
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


class CondGammaDistribution(CondVAEDistribution):
    """ A distribution over a set of conditionally independent Gamma random variables.

    We use the convention of parameterizing a Gamma distribution with concentration and rate parameters.

    Much of the implementation here has been taken from torch's own Gamma distribution.

    """

    def __init__(self, conc_f: torch.nn.Module, rate_f: torch.nn.Module):
        """ Creates a CondGammaDistribution object.

        conc_f: A module whose forward function accepts input of size n_smps*d_x and outputs concentration values in
        a tensor of size n_smps*d_y

        rate_f: A module whose forward function accepts input of size n_smps*d_x and outputs rate values in
        a tensor of size n_smps*d_y
        """

        super().__init__()
        self.conc_f = conc_f
        self.rate_f = rate_f

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp

    def form_standard_sample(self, smp):
        return smp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes conditional mean given samples.

        Args:
            x: data samples are conditioned on. Of shape n_smps*d_x.

        Returns:
            mn: mn[i,:] is the mean conditioned on x[i,:]
        """

        return self.conc_f(x)/self.rate_f(x)

    def kl(self, d_2, x: torch.tensor, smp: torch.tensor = None, return_device: torch.device = None):
        """ Computes the KL divergence between the conditional distribution represented by this object and another.

        KL divergence is computed based on the closed form formula for KL divergence between two Gamma distributions.

        Note: This function will move the conditioning data (x) to the appropriate device(s)
            so calculations can be carried out without needing to move this object or the other conditional
            distribution between devices.

        Args:
            d_2: The other conditional distribution in the KL divergence.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: This input is ignored, as KL divergence is based on a closed form formula.

            return_device: The device the calculated kl tensor should be returned to.  If None, this will
            be the device the first parameter of this object is on.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th conditioing
            input.
        """

        self_device = next(self.parameters()).device
        d_2_device = next(d_2.parameters()).device

        x_self = x.to(self_device)
        x_d_2 = x.to(d_2_device)

        if return_device is None:
            return_device = self_device

        self_conc = self.conc_f(x_self).to(return_device)
        d_2_conc = d_2.conc_f(x_d_2).to(return_device)

        self_rate = self.rate_f(x_self).to(return_device)
        d_2_rate = d_2.rate_f(x_d_2).to(return_device)

        return torch.sum((self_conc - d_2_conc) * torch.digamma(self_conc)
                         - torch.lgamma(self_conc) + torch.lgamma(d_2_conc)
                         + d_2_conc * (torch.log(self_rate) - torch.log(d_2_rate))
                         + self_conc * ((d_2_rate - self_rate) / self_rate), dim=1)

    def log_prob(self, x: torch.tensor, y: torch.Tensor) -> torch.tensor:
        """ Computes log P(y| x).

        Args:
            x: Data we condition on.  Of shape n_smps*d_x

            y: Values we desire the log probability for.  Of shape n_smps*d_y.

        Returns:
            ll: Log-likelihood of each sample. Of shape n_smps.

        """
        concentration = self.conc_f(x)
        rate = self.rate_f(x)

        return torch.sum((concentration * torch.log(rate) +
                         (concentration - 1) * torch.log(y) -
                         rate * y - torch.lgamma(concentration)), dim=1)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """ Samples from the reparameterized form of P(y|x).

          If a sample without gradients is desired, wrap the call to sample in torch.no_grad().

          Args:
              x: Data we condition on.  Of shape nSmps*d_x.

          Returns:
              y: sampled data of shape nSmps*d_y.
          """

        return torch._standard_gamma(self.conc_f(x))/self.rate_f(x)

    def std(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes conditional standard deviation given samples.  """

        return torch.sqrt(self.conc_f(x)/(self.rate_f(x)**2))

    def mode(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes conditional mode given samples. """

        return (self.conc_f(x) - 1)/(self.rate_f(x))

    def r_params(self):
        return list(self.parameters())

    def sample_to(self, smp: torch.Tensor, device: torch.device) -> torch.Tensor:
        """ Moves a sample in compact form to a given device.

        This function is provided because different distributions may return samples in arbitrary objects,
        so a custom function may be needed to move a sample to a device.

        Args:
            smp: The sample to move.

            device: The device to move the sample to.

        """
        return smp.to(device)

    def s_params(self) -> list:
        return list()


class CondGaussianDistribution(CondVAEDistribution):
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
            mn_kwargs: key-word arguments to be passed to the mean function

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
            mn_kwargs: key-word arguments to be passed to the mean function
            std_kwargs: key-word arguments to be passed to the mean function

        Returns:
            y: sampled data of shape nSmps*d_y.
        """


        mn = self.mn_f(x)
        std = self.std_f(x)

        z = torch.randn_like(std)

        return mn + z*std

    def kl(self, d_2, x: torch.tensor, smp: torch.tensor = None, return_device: torch.device = None):
        """ Computes the KL divergence between the conditional distribution represented by this object and another.

        KL divergence is computed based on the closed form formula for KL divergence between two Gaussians.

        Note: This function will move the conditioning data (x) to the appropriate device(s)
            so calculations can be carried out without needing to move this object or the other conditional
            distribution between devices.

        Args:
            d_2: The other conditional distribution in the KL divergence.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: This input is ignored, as KL divergence is based on a closed form formula.

            return_device: The device the calculated kl tensor should be returned to.  If None, this will
            be the device the first parameter of this object is on.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th conditioing
            input.
        """

        self_device = next(self.parameters()).device
        d_2_device = next(d_2.parameters()).device

        x_self = x.to(self_device)
        x_d_2 = x.to(d_2_device)

        if return_device is None:
            return_device = self_device

        mn_1 = self.mn_f(x_self).to(return_device)
        std_1 = self.std_f(x_self).to(return_device)

        mn_2 = d_2.mn_f(x_d_2).to(return_device)
        std_2 = d_2.std_f(x_d_2).to(return_device)

        d = mn_1.shape[1]

        sigma_ratio_sum = torch.sum((std_1/std_2)**2, dim=1)

        mn_diff = torch.sum(((mn_2 - mn_1)/std_2)**2, dim=1)

        log_det_1 = 2*torch.sum(torch.log(std_1), dim=1)
        log_det_2 = 2*torch.sum(torch.log(std_2), dim=1)
        log_det_diff = log_det_2 - log_det_1

        kl = .5*(sigma_ratio_sum + mn_diff + log_det_diff - d)

        return kl.squeeze()

    def form_standard_sample(self, smp):
        return smp

    def form_compact_sample(self, smp: torch.Tensor) -> torch.Tensor:
        return smp

    def sample_to(self, smp: object, device: torch.device):
        """ Moves a sample in compact form to a given device.

        This function is provided because different distributions may return samples in arbitrary objects,
        so a custom function may be needed to move a sample to a device.

        Args:
            smp: The sample to move.

            device: The device to move the sample to.

        """
        return smp.to(device)

    def r_params(self):
        return list(self.parameters())

    def s_params(self) -> list:
        return list()


class CondSpikeSlabDistribution(CondVAEDistribution):
    """ Represents a condition spike and slab distriubtion. """

    def __init__(self, d: int, spike_d: CondVAEDistribution, slab_d: CondVAEDistribution):
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


class CondMatrixProductDistribution(CondVAEDistribution):
    """ Represents conditional distributions over matrices.

    Consider a matrix, W, with N rows and M columns.  Given a tensor X with N rows and P columns of conditioning data,
    this object represents:

            P(W|X) = \prod_i=1^N P_i(W[i,:]| X[i, :]),

        where:

            P_i(W[i,:] | X[i, :]) = \prod_j=1^M P_j(W[i,j] | X[i,:]),

        where the P_j distributions are specified by the user.

    In other words, we model all entries of W as conditionally independent given X, where entries of W are modeled as
    distributed according to a different conditional distribution depending on what column they are in.


    """

    def __init__(self, dists: Sequence[CondVAEDistribution]):
        """ Creates a new CondMatrixProductDistribution object.

        Args:
            dists: dists[j] is P_j, that is the conditional distribution to use for column j.
        """
        super().__init__()
        self.dists = torch.nn.ModuleList(dists)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Computes the conditional mean of the distribtion at different samples.

        Args:
            x: A tensor of shape n_rows*d_x.

        Returns:
            mn: mn[i, :] is the mean conditioned on x[i, :]
        """

        return torch.cat([d(x) for d in self.dists], dim=1)

    def sample(self, x: torch.tensor) -> torch.tensor:
        """ Samples from a conditional distribution.

        Note: Sample is represented in compact form.  Use form_standard_sample to form
        the sample into it's matrix representation.

        Args:
            x: A tensor of shape n_rows*d_x.  x[i,:] is what row i is conditioned on.

        Returns:
            smp: smp[j] is the compact representation of the sample for column j.
        """

        return [d.sample(x) for d in self.dists]

    def form_standard_sample(self, smp: object) -> torch.tensor:
        """ Forms a standard representation of a sample from the output of sample.

        Args:
            smp: Compact representation of a sample.

        Returns:
            formed_smp: The sample represented as a matrix
        """

        return torch.cat([d.form_standard_sample(s) for s, d in zip(smp, self.dists)], dim=1)

    def form_compact_sample(self, smp: torch.tensor) -> object:
        """ Forms a compact representation of a sample given a standard representation.

        Args:
            smp: The standard representation of the sample as a matrix.

        Returns:
            formed_smp: The compact representation of the sample.
        """

        # Break up our columns of the matrix, making sure they have the right shap
        n_rows, n_cols = smp.shape
        col_smps = [smp[:, c_i].view([n_rows, 1]) for c_i in range(n_cols)]

        # Now call form standard sample on each column with the appropriate distribution
        return [d.form_standard_sample(c_s) for c_s, d in zip(col_smps, self.dists)]

    def sample_to(self, smp: object, device: torch.device):
        """ Moves a sample in compact form to a given device.

        Args:
            smp: The sample to move.

            device: The device to move the sample to.

        """
        return [d.sample_to(s, device) for s, d in zip(smp, self.dists)]

    def log_prob(self, x: torch.tensor, y: Sequence) -> torch.tensor:
        """ Computes the conditional log probability of individual rows.

        Args:
            x: Data we condition on.  Of shape n_rows*d_x

            y: Compact representation of the samples we desire the probability for.  Compact representation means the
            form of a sample as output by the sample() function.

        Returns:
            ll: Conditional log probability of each row. Of shape n_rows.
        """

        # Calculate the log-likelihood of each entry in the matrix
        n_rows = x.shape[0]
        entry_ll = torch.cat([d.log_prob(x, c_s).view([n_rows, 1]) for c_s, d in zip(y, self.dists)], dim=1)
        return torch.sum(entry_ll, dim=1)

    def kl(self, d_2, x: torch.tensor, smp: Sequence = None, return_device: torch.device = None):
        """ Computes the KL divergence between this object and another CondMatrixProductDistribution conditioned on input.

        This function overrides the default kl function of CondVAEDistribution so that the KL divergence is
        computed between distributions for the same column and then summed up. This is still mathematically
        correct, but if the distributions for the columns also override kl, then distribution specific kl
        calculations (perhaps analytical calculations) can be carried out.

        Args:
            d_2: The other conditional distribution in the KL divergence.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: An set samples of shape n_smps*d_y. smp[i,:] should be drawn this objects distribution.  This input is
            provided because some distributions for the columns may not analytically compute KL divergence.

            return_device: The device the calculated kl tensor should be returned to.  If None, this will
            be the device the first parameter of this object is on.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th sample.

        """

        n_cols = len(self.dists)

        if smp is None:
            smp = [None]*len(self.dists)

        kl = self.dists[0].kl(d_2.dists[0], x=x, smp=smp[0], return_device=return_device)
        for c_i in range(1, n_cols):
            kl += self.dists[c_i].kl(d_2.dists[c_i], x=x, smp=smp[c_i], return_device=return_device)

        return kl

    def r_params(self):
        return list(self.parameters())

    def s_params(self) -> list:
        return list()


class CondGaussianMatrixProductDistribution(CondMatrixProductDistribution):
    """ Represents conditional Gaussian distributions over matrices.

    Consider a matrix, W, with N rows and M columns.  Given a tensor X with N rows and P columns of conditioning data,
    this object represents:

            P(W|X) = \prod_i=1^N P_i(W[i,:]| X[i, :]),

        where:

            P_i(W[i,:] | X[i, :]) = \prod_j=1^M P_j(W[i,j] | X[i,:]),

        where the P_j distributions are conditional Gaussian distributions specified by the user.

    In other words, we model all entries of W as conditionally independent given X, where entries of W are modeled as
    distributed according to a different conditional Gaussian distribution depending on what column they are in.

    This objects extends CondMatrixProductDistribution, and it's main purpose is to allow KL divergences to be
    computed not only between itself and another CondMatrixProductDistribution but also a CondGaussianDistribtion when
    both distributions are over matrices of the same shape.

    """

    def __init__(self, dists: Sequence[CondGaussianDistribution]):

        for d in dists:
            if not isinstance(d, CondGaussianDistribution):
                raise(TypeError('All distributions must be CondGaussianDistribution objects.'))

        super().__init__(dists=dists)

    def kl(self, d_2, x: torch.tensor, smp: torch.tensor = None, return_device: torch.device = None):
        """ Computes the KL divergence between the conditional distribution represented by this object and another.

        The second distribtion can be either another CondMatrixProductDistribution or a CondGaussianDistribution over
        matrices of the same size this distribution is over.

        KL divergence is computed based on the closed form formula for KL divergence between two Gaussians.

        Note: This function will move the conditioning data (x) to the appropriate device(s)
            so calculations can be carried out without needing to move this object or the other conditional
            distribution between devices.

        Args:
            d_2: The other conditional distribution in the KL divergence.

            x: A tensor of shape n_smps*d_x.  x[i,:] is what sample i is conditioned on.

            smp: This input is ignored, as KL divergence is based on a closed form formula.

            return_device: The device the calculated kl tensor should be returned to.  If None, this will
            be the device the first parameter of this object is on.

        Returns:
            kl: Of shape n_smps.  kl[i] is the KL divergence between the two distributions for the i^th conditioning
            input.
        """

        if isinstance(d_2, CondGaussianMatrixProductDistribution):
            return super().kl(d_2=d_2, x=x, smp=smp, return_device=return_device)
        elif isinstance(d_2, CondGaussianDistribution):

            self_device = next(self.parameters()).device
            d_2_device = next(d_2.parameters()).device

            # Make (possible) copies of conditioning data on each device we need it
            x_self = x.to(self_device)
            x_d_2 = x.to(d_2_device)

            if return_device is None:
                return_device = self_device

            mn_1 = torch.cat([d.mn_f(x_self).to(return_device) for d in self.dists], dim=1)
            std_1 = torch.cat([d.std_f(x_self).to(return_device) for d in self.dists], dim=1)

            mn_2 = d_2.mn_f(x_d_2).to(return_device)
            std_2 = d_2.std_f(x_d_2).to(return_device)

            if not mn_1.shape == mn_2.shape:
                raise(ValueError('Cannot compute KL divergence between distributions over matrices of different shapes.'))

            d = mn_1.shape[1]

            sigma_ratio_sum = torch.sum((std_1 / std_2) ** 2, dim=1)

            mn_diff = torch.sum(((mn_2 - mn_1) / std_2) ** 2, dim=1)

            log_det_1 = 2 * torch.sum(torch.log(std_1), dim=1)
            log_det_2 = 2 * torch.sum(torch.log(std_2), dim=1)
            log_det_diff = log_det_2 - log_det_1

            kl = .5 * (sigma_ratio_sum + mn_diff + log_det_diff - d)

            return kl.squeeze()
        else:
            raise(TypeError('d_2 must be either a CondGaussianMatrixProductDistribution or a CondGaussianDistribution.'))


class MatrixGammaProductDistribution(CondMatrixProductDistribution):
    """ Represents a distribution over matrices where each entry is pulled iid from a separate Gamma distribution.

    For a matrix, W, with N rows and M columns, we model:

        P(W) = \prod_i=1^N \prod_j=1^M P_ij(W[i,j]),

    where P_ij is a Gamma distribution with concentration parameter \alpha_ij and rate parameter \beta_ij.

    Note: This function extends CondMatrixProductDistribution, allowing this distribution to be used in
    code where conditional distributions are required, so that the resulting "conditional distributions"
    are the same irrespective of conditioning input.

    """

    def __init__(self, shape: Sequence[int], conc_lb: float = 1.0, conc_ub: float = 1000.0, conc_iv: float = 10.0,
                 rate_lb: float = .001, rate_ub: float = 1000.0, rate_iv: float = 10.0):
        """ Creates a new MatrixGammaProductDistribution object.

        Args:

            shape: The shape of matrices this represents distributions over.

            conc_lb: The lower bound that concentration parameters can take on

            conc_ub: The upper bound that concentration parameters can take on

            conc_iv: The initial value for concentration parameters.  All distributions will be initialized to have the
            same initial values.

            rate_lb: The lower bound that rate parameters can take on

            rate_ub: The upper bound that rate parameters can take on

            rate_iv: The initial value for rate parameters.  All distributions will be initialized to have the
            same initial values.
        """
        n_rows, n_cols = shape
        col_dists = [None]*n_cols
        for c_i in range(n_cols):
            conc_f = IndSmpConstantBoundedFcn(n=n_rows, lower_bound=conc_lb, upper_bound=conc_ub, init_value=conc_iv,
                                              check_sizes=False)
            rate_f = IndSmpConstantBoundedFcn(n=n_rows, lower_bound=rate_lb, upper_bound=rate_ub, init_value=rate_iv,
                                              check_sizes=False)
            col_dists[c_i] = CondGammaDistribution(conc_f=conc_f, rate_f=rate_f)

        # Create the object
        super().__init__(dists=col_dists)
        self.n_rows = n_rows

    #def forward(self, x: torch.Tensor = None):
    #    """ Overwrites parent forward so x does not have to be provided. """
    #    return super().forward(x=torch.zeros([self.n_rows, 1]))

    #def sample(self, x: torch.Tensor = None) -> list:
    #    """ Overwrites parent sample so x does not have to be provided. """
    #    return super().sample(x=torch.zeros([self.n_rows, 1]))

    #def log_prob(self, x: torch.Tensor = None, y: Sequence = None) -> torch.Tensor:
    #    """ Overwrites parent log_prob so x does not have to be provided.

    #    Raises:
    #        ValueError: If y is None.
    #   """
    #    if y is None:
    #        raise(ValueError('y value cannot be none'))
    #
    #    return super().log_prob(x=torch.zeros([self.n_rows, 1]), y=y)


class MatrixFoldedNormalProductDistribution(CondMatrixProductDistribution):
    """ Represents a distribution over matrices where each entry is pulled iid from a Folded Normal distribution.

    For a matrix, W, with N rows and M columns, we model:

        P(W) = \prod_i=1^N \prod_j=1^M P_ij(W[i,j]),

    where P_ij is a Folded Normal distribution with parameters \mu_ij and \sigma_ij.

    Note: This function extends CondMatrixProductDistribution, allowing this distribution to be used in
    code where conditional distributions are required, so that the resulting "conditional distributions"
    are the same irrespective of conditioning input.

    """

    def __init__(self, shape: Sequence[int], mu_lb: float = 0.0, mu_ub: float = 10.0, mu_iv: float = 1.0,
                 sigma_lb: float = .001, sigma_ub: float = 10.0, sigma_iv: float = 1.0):
        """ Creates a new MatrixGammaProductDistribution object.

        Args:

            shape: The shape of matrices this represents distributions over.

            mu_lb: The lower bound that mu parameters can take on

            mu_ub: The upper bound that mu parameters can take on

            mu_iv: The initial value for mu parameters.  All distributions will be initialized to have the
            same initial values.

            sigma_lb: The lower bound that sigma parameters can take on

            sigma_ub: The upper bound that sigma parameters can take on

            sigma_iv: The initial value for sigma parameters.  All distributions will be initialized to have the
            same initial values.
        """
        n_rows, n_cols = shape
        col_dists = [None]*n_cols
        for c_i in range(n_cols):
            mu_f = IndSmpConstantBoundedFcn(n=n_rows, lower_bound=mu_lb, upper_bound=mu_ub, init_value=mu_iv)
            sigma_f = IndSmpConstantBoundedFcn(n=n_rows, lower_bound=sigma_lb, upper_bound=sigma_ub,
                                               init_value=sigma_iv)
            col_dists[c_i] = CondFoldedNormalDistribution(mu_f=mu_f, sigma_f=sigma_f)

        # Create the object
        super().__init__(dists=col_dists)
        self.n_rows = n_rows

    def forward(self, x: torch.Tensor = None):
        """ Overwrites parent forward so x does not have to be provided. """
        return super().forward(x=torch.zeros([self.n_rows, 1]))

    def sample(self, x: torch.Tensor = None) -> list:
        """ Overwrites parent sample so x does not have to be provided. """
        return super().sample(x=torch.zeros([self.n_rows, 1]))

    def log_prob(self, x: torch.Tensor = None, y: Sequence = None) -> torch.Tensor:
        """ Overwrites parent log_prob so x does not have to be provided.

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise(ValueError('y value cannot be none'))

        return super().log_prob(x=torch.zeros([self.n_rows, 1]), y=y)


class MatrixGaussianProductDistribution(CondGaussianMatrixProductDistribution):
    """ Represents a distribution over matrices where each entry is pulled iid from a separate Gaussian distribution.

    For a matrix, W, with N rows and M columns, we model:

        P(W) = \prod_i=1^N \prod_j=1^M P_ij(W[i,j]),

    where P_ij is a Gaussian distribution with mean mu_ij and standard deviation std_ij.

    Note: This function extends CondMatrixProductDistribution, allowing this distribution to be used in
    code where conditional distributions are required, so that the resulting "conditional distributions"
    are the same irrespective of conditioning input.

    """

    def __init__(self, shape: Sequence[int], mn_mn: float = 0.0, mn_std: float = .01,
                 std_lb: float = .000001, std_ub: float = 10.0, std_iv: float = .01):
        """ Creates a new MatrixGaussianProductDistribution.

        Args:
            shape: The shape of matrices this represents distributions over.

            mn_mn, std_mn: The mean and standard deviation to use when generating random initial values for the
            mean distribution for each entry.

            std_lb, std_ub, std_iv: lower & upper bounds for standard deviation values and the initial value
            for the standard deviation for the distribution for each entry.

        """

        # Generate the distributions for each column
        n_rows, n_cols = shape
        col_dists = [None]*n_cols
        for c_i in range(n_cols):
            mn_f = IndSmpConstantRealFcn(n=n_rows, init_mn=mn_mn, init_std=mn_std, check_sizes=False)
            std_f = IndSmpConstantBoundedFcn(n=n_rows, lower_bound=std_lb, upper_bound=std_ub, init_value=std_iv,
                                             check_sizes=False)
            col_dists[c_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        # Create the object
        super().__init__(dists=col_dists)
        self.n_rows = n_rows

    def initialize(self, mn_mn: float = 0.0, mn_std: float = .01, std_v: float = .01):
        """ Initializes parameters of the distribution.

        Args:
            mn_mn, mn_std: The mean and standard deviation for the distribution values for the mean are drawn from

            std_v: The value to set the standard deviation to everywhere
        """
        for d in self.dists:
            torch.nn.init.normal_(d.mn_f.f.vl, mean=mn_mn, std=mn_std)

            std_device = d.std_f.f.lower_bound.device
            lower_bound = d.std_f.f.lower_bound.cpu().numpy()
            upper_bound = d.std_f.f.upper_bound.cpu().numpy()
            init_v = np.arctanh(2*(std_v - lower_bound)/(upper_bound - lower_bound) - 1)
            init_v = torch.Tensor(init_v)
            init_v = init_v.to(std_device)
            d.std_f.f.v.data = init_v

    #def forward(self, x: torch.Tensor = None) -> torch.Tensor:
    #    """ Overwrites parent forward so x does not have to be provided. """
    #    return super().forward(x=torch.zeros([self.n_rows, 1]))

    #def sample(self, x: torch.Tensor = None) -> list:
    #    """ Overwrites parent sample so x does not have to be provided. """
    #    return super().sample(x=torch.zeros([self.n_rows, 1]))

    #def log_prob(self, x: torch.Tensor = None, y: Sequence = None) -> torch.Tensor:
    #    """ Overwrites parent log_prob so x does not have to be provided.
    #
    #    Raises:
    #        ValueError: If y is None.
    #    """
    #    if y is None:
    #        raise(ValueError('y value cannot be none'))

    #    return super().log_prob(x=torch.zeros([self.n_rows, 1]), y=y)


class CondMatrixHypercubePrior(CondGaussianMatrixProductDistribution):
    """ Extends CondGaussianMatrixProductDistribution so the distribution for each column is a Gaussian with mean and standard
    deviation functions which are sums of tiled hypercube basis functions.

    Specifically, For a matrix, W, under a CondMatrixProductDistribution, we model:

        W[i,j] ~ P_j(W[i,j] | X[i,:]).

    Here, we specify that P_j is a conditional Gaussian distribution with mean given by m(X[i,:]) and standard
    deviation by s(X[i,:]). Specifically, m() is a SumOfTiledHyperCubeBasisFcns function and s() is an exponentiated
    SumOfTiledHyperCubeBasisFcns function plus an offset.

    """

    def __init__(self, n_cols: int, mn_hc_params: dict, std_hc_params: dict, min_std: float,
                 mn_init: float = 0.0, std_init: float = .01):
        """ Creates a CondMatrixHypercubePrior object

        Args:
            n_cols: The number of columns in the matrices we represent distributions over.

            mn_hc_params: A dictionary with parameters for passing into the init() function of
            SumOfTiledHyperCubeBasisFcns when creating the hypercube function for the mean function for each P_j.

            std_hc_params: A dictionary with parameters for passing into the init() function of
            SumOfTiledHyperCubeBasisFcns when creating the hypercube function which will be exponentiated and offset
            to form the final standard deviation function for each P_j.

            min_std: The min standard deviation any P_j can take on.

            mn_init: The initial value for the mean function. The mean will take on this value everywhere.

            std_init: The initial value for the standard deviation function.  The standard deviation will take
            on this value everywhere. Must be greater than min_std

        Raises:
            ValueError: If std_init is not greater than min_std.

        """


        if std_init <= min_std:
            raise(ValueError('std_init must be greater than min_std'))

        # Form each P_j
        col_dists = [None]*n_cols
        for c_i in range(n_cols):

            # Create mean function, setting it's initial value
            mn_f = SumOfTiledHyperCubeBasisFcns(**mn_hc_params)

            # Create standard deviation function, setting it's initial value
            std_hc_f = SumOfTiledHyperCubeBasisFcns(**std_hc_params)
            std_f = torch.nn.Sequential(std_hc_f, FixedOffsetExp(min_std))

            # Create the distribution for the column
            col_dists[c_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        super().__init__(dists=col_dists)
        self.n_cols = n_cols
        self.mn_hc_params = mn_hc_params
        self.std_hc_params = std_hc_params
        self.min_std = min_std

        self.set_mn(mn_init)
        self.set_std(std_init)

    def increase_std(self, f: float):
        """ Increases the standard deviation by a factor which is approximately log(f).

        Args:
            f: The factor to increase standard deviation by.
        """
        for d in self.dists:
            d.std_f[0].b_m.data[:] = d.std_f[0].b_m.data[:] + np.log(f)

    def set_mn(self, v: float):
        """ Set the mean to a single value everyhwere.

        Args:

            v: The value to set the mean to
        """

        for c_i in range(self.n_cols):
            mn_hc_f = self.dists[c_i].mn_f
            n_basis_fcns_per_mn_cube = np.cumprod(self.mn_hc_params['n_div_per_hc_side_per_dim'])[-1]
            mn_cube_vl = v / n_basis_fcns_per_mn_cube
            mn_hc_f.b_m.data = mn_cube_vl * torch.ones_like(mn_hc_f.b_m.data)

    def set_std(self, v: float):
        """ Sets the standard deviation to a single value everywhere.

        Args:

            v: The value to set the standard deviation to

        """
        for c_i in range(self.n_cols):
            std_hc_f = self.dists[c_i].std_f[0]
            n_basis_fcns_per_std_cube = np.cumprod(self.std_hc_params['n_div_per_hc_side_per_dim'])[-1]
            std_cube_vl = np.log(v - self.min_std) / n_basis_fcns_per_std_cube
            std_hc_f.b_m.data = std_cube_vl * torch.ones_like(std_hc_f.b_m.data)


class GroupCondMatrixHypercubePrior(CondGaussianMatrixProductDistribution):
    """ Extends CondGaussianMatrixProductDistribution so the distribution for each column is a Gaussian with
    mean and standard deviation functions that depend on groups of properties.

    Specifically, For a matrix, W, under a CondMatrixProductDistribution, we model:

        W[i,j] ~ P_j(W[i,j] | X[i,:]).

    Here, we specify that P_j is a conditional Gaussian distribution with mean given by m(X[i,:]) and standard
    deviation by s(X[i,:]). For m(), we specify

        m(X[i,:]) = s_mn*tanh( \sum_{ind_j \in inds} f^mn_j(X[i, ind_j]) ) + o_mn,

    where each f_j() is a SumOfTiledHyperCubeBasisFcns function.

    For s(), we specify

        s(X[i,:]) = exp( \sum_{ind_j \in inds} f^std_j(X[i, ind_j]) ) + min_std,

    where min_std is a fixed, small offset ensuring s() stays strictly positive.

    """
    def __init__(self, n_cols: int, group_inds: Sequence[Sequence[int]],
                 mn_hc_params: Sequence[dict], std_hc_params: Sequence[dict],
                 min_std: float, mn_init: float, std_init: float,
                 tanh_init_opts: dict = None):
        """ Creates a new GroupCondMatrixHypercubePrior object.

        Args:
            n_cols: The number of columns in the matrices we represent distributions over.

            group_inds: group_inds[j] are the indices into the dimensions of X for properties for group j

            mn_hc_params: mn_hc_params[j] is a dictionary with parameters for passing into the init() function of
            SumOfTiledHyperCubeBasisFcns when creating the hypercube function for f^mn_j.

            std_hc_params: std_hc_params[j] is a dictionary with parameters for passing into the init() function
            of SumOfTiledHyperCubeBasisFcns when creating the hypercube function for f^std_j.

            min_std: The min standard deviation any P_j can take on.

            mn_init: The initial value for the mean function. The mean will take on this value everywhere.

            std_init: The initial value for the standard deviation function.  The standard deviation will take
            on this value everywhere. Must be greater than min_std

            tanh_init_opts: Dictionary of additional options when initializing the Tanh module for
            the mean function for each mode. If None, no options will be passed

        Raises:
            ValueError: If std_init is not greater than min_std.
        """

        if std_init <= min_std:
            raise(ValueError('std_init must be greater than min_std'))

        if tanh_init_opts is None:
            tanh_init_opts = dict()

        # Make sure group inds are of the appropriate type
        group_inds = [torch.tensor(inds, dtype=torch.int64) for inds in group_inds]

        n_prop_spaces = len(group_inds)

        dists = [None]*n_cols
        for c_i in range(n_cols):

            # Setup mn function
            mn_hc_fcns = [SumOfTiledHyperCubeBasisFcns(**params) for params in mn_hc_params]
            mn_fcn = torch.nn.Sequential(SCC(group_inds=group_inds, group_modules=mn_hc_fcns),
                                         SumAlongDim(dim=1), Tanh(d=1, **tanh_init_opts))

            # Set initial value of mean function
            mn_fcn[2].o.data[:] = mn_init

            # Setup std function
            std_hc_fcns = [SumOfTiledHyperCubeBasisFcns(**params) for params in std_hc_params]
            std_fcn = torch.nn.Sequential(SCC(group_inds=group_inds, group_modules=std_hc_fcns),
                                          SumAlongDim(dim=1), FixedOffsetExp(o=min_std))

            # Set initial value of std_hc_fcns
            for p_i, fcn in enumerate(std_hc_fcns):
                div_f = n_prop_spaces*np.prod(mn_hc_params[p_i]['n_div_per_hc_side_per_dim'])
                fcn.b_m.data[:] = np.log(std_init - min_std)/div_f

            dists[c_i] = CondGaussianDistribution(mn_f=mn_fcn, std_f=std_fcn)

            super().__init__(dists=dists)


class DistributionPenalizer(torch.nn.Module):
    """ A base class for creating distribution penalizer objects.

    The main idea behind a penalizer object (vs. just applying a penalizer function) is that the ways we may
    want to penalize a distribution may require keeping track of some penalty parameters (e.g., a set of locations
    where we want to sample a distribution at).  Some of these parameters could even be optimizable.  Because of this,
    we introduce this concept of penalizer objects which are torch modules, so we can keep track of these parameters,
    easily move them between devices, etc...

    """

    def __init__(self):
        """ Creates a DistributionPenalizer object. """
        super().__init__()

    def check_point(self) -> dict:
        """ Returns a dictionary of parameters for the penalizer that should be saved in a check point.

        The idea is that for the purposes of creating a check point, we can save memory by only logging the
        important parameters of a penalizer.
        """
        raise(NotImplementedError)

    def get_marked_params(self, key: str):
        """ Returns all parameters marked with the key string.

        Penalizers must associate each parameter with a unique key (e.g., fast_learning_rate_params). Each
        parameter should be associated with only one key (though multiple parameters can use the same key).  This
        function will return a list of parameters associated with the requested key.  If no parameters match the
        key an empty list should be returned.

        """
        raise(NotImplementedError)

    def list_param_keys(self):
        """ Returns a list of keys associated with parameters.

        Returns:
            keys: The list of keys.
        """
        raise(NotImplementedError)

    def penalize(self, d: CondVAEDistribution) -> torch.Tensor:
        """ Calculates a penalty over a distribution.

        Args:
            d: The distribution to penalize

        Returns:
            penalty: The scalar penalty
        """
        raise(NotImplementedError)


class ColumnMeanClusterPenalizer(DistributionPenalizer):
    """ Penalizes the mean of a conditional distribution over matrices to encouraging clustering of column values.

    Clustering here means clustering of values given what they are conditioned on.

    In particular, we work with conditional distributions over matrices M \in R^{n \times p} conditioned on
    input X \in R^{n \times q}, where each row of M is associated with the corresponding row of X.  Our goal is
    to encourage large values in each column of M to be assoicated with values in X that are close in space.

    We achieve this by:

        1) Keeping track of a "center" parameter for each column c_j \in R^{1 \times q} and "scale"
        parameter s_j \in R^{1 \times q} for each column j \in [1, p].  These parameters are learnable (but the
        user can chose to fix the scales).

        2) Let E_j be the expected value of column j conditioned on X.  We compute the cluster penalty for
        column j as: k_j = \sum_i w_i*d_i, where w_i is the absolute value of E_j[j] after E_j has been normalized
        to have a length of 1 and d_i is the square of scaled distance from c_j defined as
        d_i = \sum_k=1^q ((X[i, k] - c_j[k])/s_j[k])**2.  To guard against division by zero, small offsets are
        added as needed in the calculations.

        3) The penalty can be made arbitrarily small by driving the scales to infinity.  To prevent this,
        we calculate a term p = \sum_j \sum_q s_j[q]**2

        4) The final penalty is scale_penalty*p + \sum_j k_j

    """
    def __init__(self, init_ctrs: torch.Tensor, x:torch.Tensor, init_scales: torch.Tensor = None,
                 scale_weight: float = None):
        """ Creates a new ColumnMeanClusterPenalizer object.

        Args:

            init_ctrs: Initial centers for each column.  Of shape [n_cols, x_dim]

            x: The points at which we evaluate the mean of the distribution. Of shape [n_pts, x_dim].

            init_scales: Initial scales for each column. Of shape [n_cols, x_dim]. If None, initial scales will be set
            to 1 for all dimensions and modes.

            scale_weight: The weight to apply to the scale penalty if learning scales.  If None, the scales will be be
            fixed at their init values and not learned.

        """
        super().__init__()

        n_modes, n_cols = init_ctrs.shape

        # Setup centers
        self.n_modes = n_modes  # We refer to columns as modes in the code
        self.col_ctrs = torch.nn.Parameter(init_ctrs)
        self.register_buffer('x', x)

        self.last_weight_pen = 0
        self.last_scale_pen = 0

        # Setup weights
        self.scale_weight = scale_weight
        if scale_weight is not None: # Indicates we want to treat weights as a parameter
            if init_scales is None:
                init_scales = torch.ones([n_modes, n_cols])
            self.scales = torch.nn.Parameter(init_scales)
        else: # Indicates we will fix weights at initial values and not learn them
            self.register_buffer('scales', torch.tensor(init_scales))

    def check_point(self):
        """ Returns a dictionary with a copy of key parameters of the penalizer.

         Returns:
             params: A dictionary with the following keys:
                col_ctrs: The value of column centers
                scales: The value of the scales
                last_weight_pen: The value of the last weight penalty computed with the penalizer
                last_scale_pen: The value of the last scale penalty computed with the penalizer
         """

        col_ctrs_copy = copy.deepcopy(self.col_ctrs)
        col_ctrs_copy = col_ctrs_copy.detach().cpu().numpy()

        scales_copy = copy.deepcopy(self.scales)
        scales_copy = scales_copy.detach().cpu().numpy()

        return {'col_ctrs': col_ctrs_copy, 'scales': scales_copy,
                'last_weight_pen': self.last_weight_pen, 'last_scale_pen': self.last_scale_pen}

    def get_marked_params(self, key: str):
        """ Returns parameters that should be assigned fast and slow learning rates.

        Args:
            key: The type of parameters that should be returned.  'fast' will return parameters that should be trained
            with fast learning rates; 'slow' will return parameters that should be trained with slow training weights

        Returns:
            params: The list of parameters matching the key
        """
        if key == 'fast':
            return [self.col_ctrs]
        elif key == 'slow':
            if self.scale_weight is not None:
                return [self.scales]
        else:
            return []

    def list_param_keys(self):
        """ Returns the list of keys associated with parameters.

        Returns:
            keys: The keys
        """
        if self.scale_weight is not None:
            return ['fast', 'slow']
        else:
            return ['fast']

    def penalize(self, d: CondVAEDistribution) -> torch.Tensor:
        """ Calculates the penalty for a distribution. """

        # Move the penalizer to the same device as the distribution
        d_device = next(d.parameters()).device
        self.to(d_device)

        n_x = self.x.shape[0]
        mn = d(self.x)
        penalty = 0
        for m_i in range(self.n_modes):

            mn_i = torch.abs(mn[:, [m_i]])
            norm_vl = torch.sum(mn_i**2) + .000001
            dist_i_scaled_sq = torch.sum(((self.x - self.col_ctrs[m_i, :])**2)/((self.scales[m_i, :])**2 + .001), dim=1)
            weighted_dist_i = (torch.squeeze(mn_i)/torch.sqrt(norm_vl))*dist_i_scaled_sq
            penalty += torch.sum(weighted_dist_i)

        self.last_weight_pen = penalty.detach().cpu().numpy()

        # Penalize for scale
        if self.scale_weight is not None and self.scale_weight != 0:
            penalty += self.scale_weight*(torch.sum(self.scales**2))

        self.last_scale_pen = penalty.detach().cpu().numpy() - self.last_weight_pen

        return penalty

    def __str__(self):
        """ Returns a string of the current state of the penalizer, including the last weight and scale penalty values. """
        return ('Weight penalty: ' + str(self.last_weight_pen) +
                '\n Scale penalty: ' + str(self.last_scale_pen))# +
                #'\n Centers: \n' + str(self.col_ctrs.t()) + '\n Scales: \n' + str(self.scales.t()))


def gen_columns_mean_cluster_penalizer(n_cols: int, dim_ranges: np.ndarray, n_pts_per_dim: Sequence[int],
                                       n_ctrs_per_dim: Sequence[int],
                                       init_scale: float = 100.0,
                                       scale_weight: float = 10.0,
                                       penalizer_pts: torch.Tensor = None) -> ColumnMeanClusterPenalizer:
    """ Generates a columns mean cluster penalizer for a conditional distribution over matrices.

    Args:
        n_cols: The number of columns in the matrices the conditional distribution is over.

        dim_ranges: dim_ranges[:,0] are the starting values for each dimension of the data the distribtion is
        conditioned on and dim_ranges[:,1] are the ending values

        n_ctrs_per_dim: When generating the initial center points, we will lay them out evenly on a grid.  This is
        the number of points on the grid in each dimension.

        n_pts_per_dim: The number of sample points to generate per dimension.  The final sample points will
        be a grid sampled at this many points per dimension within the range of dimensions specified by dim_ranges.

        init_scale: The value that for the initial scales of the penalizer for all modes and dimensions.

        scale_weight: The scale weight for the penalizer.

        penalizer_pts: A tensor of points to penalize at.  If None, one will be created based on dim_ranges and
        n_pts_per_dim.  Using this input is useful if creating multiple penalizers that all use the same penalizer
        points, so that they can all reference the same list of points and duplicate lists of poitns do not have to be
        created to save memory.

    Returns:
        p: The generated penalizer.
    """

    # Generate points we penalize at
    if penalizer_pts is None:
        penalizer_grid_limits = copy.deepcopy(dim_ranges)
        penalizer_grid_limits[:, 0] = penalizer_grid_limits[:, 0] + .001
        penalizer_grid_limits[:, 1] = penalizer_grid_limits[:, 1] - .001

        penalizer_n_grid_pts = 2*np.asarray(n_pts_per_dim)

        penalizer_pts = torch.tensor(list_grid_pts(grid_limits=penalizer_grid_limits,
                                               n_pts_per_dim=penalizer_n_grid_pts).astype(np.float32))

    init_ctrs = torch.Tensor(list_grid_pts(grid_limits=dim_ranges, n_pts_per_dim=n_ctrs_per_dim))

    # Generate initial scales
    init_scales = init_scale*torch.ones([n_cols, 3])

    return ColumnMeanClusterPenalizer(init_ctrs=init_ctrs, x=penalizer_pts, scale_weight=scale_weight,
                                      init_scales=init_scales)