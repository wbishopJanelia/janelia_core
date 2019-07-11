""" Tools for generating simulated latent regression models. """

from typing import Sequence

import numpy as np
import torch

def generate_simple_bump_modes_scenario(n_subjects, n_modes: int, n_neuron_range: Sequence[int],
                                        bump_std_range: Sequence[int] = [.2, .4], prior_std: float = .1,
                                        n_dims: int = 2):

    """ Generates a set of models with modes pulled from bump shaped spatial priors.

    By "bump shaped prior" we mean that the conditional distribution for a neuron's loading in a particular mode
    given it's position in space is Gaussian with a conditional mean which is shaped like a bump in space. In this
    simple scenario, we assume the standard deviation of the conditional distributions is the same irrespective of a
    neuron's location in space.

    Other than having modes which are pulled from Gaussian bump-shaped priors, this scenario is as simple as possible.

    Specifically:

        (1) There is only one group of variables.
        (2) There are no direct connections.
        (3) We use an identity map in the latent space.
        (4) Neurons for each subject are uniformly distributed in the unit hypercube.

    Args:
        n_subjects: The number of subject models to generate

        n_modes: The number of modes each model should have

        n_neuron_range: The number of neurons each subject has will be pulled uniformly from this range

        bump_std_range: We use a Gaussian kernel for the bump functions which form the conditional means for each mode.
        These kernel functions will be axis aligned with a certain standard deviation along each axis.  The standard
        deviation values along each axis will be pulled uniformly from this range.

        prior_std: The standard deviation of each prior conditional distribution.

        n_dims: The number of spatial dimensions neurons are arranged in.

    """

    # Generate the spatial priors for p & u mode loadings
    bump_std_int_width = bump_std_range[1] - bump_std_range[0]

    p_prior_mode_dists = [None]*n_modes
    for m_i in range(n_modes):
        mode_stds = np.random.rand(n_dims)*bump_std_int_width + bump_std_range[0]
        print(mode_stds)


    # Generate the spatial priors for u mode loadings


    pass
