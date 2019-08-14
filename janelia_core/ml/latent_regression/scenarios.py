""" Tools for generating simulated latent regression models. """

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import torch

from janelia_core.ml.datasets import TimeSeriesDataset
from janelia_core.ml.extra_torch_modules import FixedOffsetExp
from janelia_core.ml.extra_torch_modules import IndSmpConstantBoundedFcn
from janelia_core.ml.extra_torch_modules import IndSmpConstantRealFcn
from janelia_core.ml.extra_torch_modules import SumOfTiledHyperCubeBasisFcns
from janelia_core.ml.latent_regression.group_maps import ConcatenateMap
from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.latent_regression.vi import SubjectVICollection
from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.torch_distributions import CondMatrixProductDistribution
from janelia_core.ml.utils import torch_mod_to_fcn
from janelia_core.visualization.image_generation import generate_image_from_fcn


class GaussianBumpFcn(torch.nn.Module):
    """ A Gaussian bump function with fixed parameters.

     The bump function will be axis-aligned but can have different standard deviations along each axis.
     """

    def __init__(self, ctr: torch.Tensor, std: torch.Tensor, peak_vl: float):
        """ Creates a new GaussianBump module.

        Args:
            ctr: A 1-d tensor giving the center location of the bump.

            std: A 1-d tensor giving the standard deviation of the bump along each axis.

            peak_vl: The value at the peak of the bump.
        """

        super().__init__()

        self.ctr = torch.nn.Parameter(ctr, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)
        self.peak_vl = torch.nn.Parameter(torch.Tensor([peak_vl]), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """ Computes input from output.

        Args:
            x: Input.  Each row is a sample.

        Returns:
            y: Output.  Each row corresponds to a transformed input sample.
        """

        n_smps = x.shape[0]

        x_ctr = x - self.ctr
        x_ctr_scaled = x_ctr/self.std

        if len(x_ctr_scaled.shape) > 1:
            x_dist = torch.sum(x_ctr_scaled**2, dim=1)
        else:
            x_dist = x_ctr_scaled**2

        return self.peak_vl*torch.exp(-1*x_dist).reshape([n_smps, 1])


class ConstantFcn(torch.nn.Module):
    """ Represents a function which is constant everywhere with fixed parameters. """

    def __init__(self, c: torch.Tensor):
        """ Creates a ConstantFcn object.

        Args:
            c: Constant value of function.  A 1-d array with length equal to dimensionality of output.
        """
        super().__init__()

        self.c = torch.nn.Parameter(c, requires_grad=False)
        self.n_dims = len(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Computes output from input.

        Args:
            x: Input of shape n_smps*d_x

            y: Output of shape n_smps*d_y.  (Each row will be equal to the constant value of the function.)
        """

        n_smps = x.shape[0]
        return self.c.unsqueeze(0).expand(n_smps, self.n_dims)


class IdentityS(torch.nn.Module):
    """ Module which just passes through input. """

    def forward(self, x):
        return x


class BumpInputWithRecursiveDynamicsScenario():
    """ Objects for generating models and data where we record neurons from multiple subjects receiving stimuli.

    We simulate recording neural activity from multiple subjects while they are presented stimuli.  There are a set of
    inputs that drive different patterns of activity across the population.  Activity across the population is then read
    out and in turn drive other sets of patterns (internal dynamics).

    Neurons are characterized by their position in a unit hypercube (number of dimensions is user set).

    The patterns of activity that is driven be each stimulus are determined by the loadings of a set of modes, one mode
    for each stimulus.  The loadings for neurons for each mode is a probabilistic function of their position.
    Specifically, we select loadings for modes for each subject from bump shaped priors.  Similarly, the modes which
    determine the internal dynamics of the population also come from bump shaped priors.

    By "bump shaped prior" we mean that the conditional distribution for a neuron's loading in a particular mode
    given it's position in space is Gaussian with a conditional mean which is shaped like a bump in space. In this
    simple scenario, we assume the standard deviation of the conditional distributions is the same irrespective of a
    neuron's location in space.

    We select the p modes of the internal dynamics to be roughly aligned with the u modes of the stimuli, to ensure
    that stimulus activity is propagated into the internal dynamics of the population.

    The logic for using this object is as follows:

        1) Initialize a scenario - this creates the priors and the subject models

        2) Simulate data for the scenario using the generate_data function

    """

    def __init__(self, n_subjects, n_modes: int, n_neuron_range: Sequence[int], prior_std: float = .02,
                 bump_max_range: Sequence[float] = [.09, .2], bump_std_range: Sequence[int] = [.2, .4],
                 noise_range: Sequence[float] = [.1, .2], n_dims: int = 2, pos_neg_modes: bool = True,
                 input_bump_ctrs: np.ndarray = None, input_bump_stds: np.ndarray = None,
                 input_bump_gains: np.ndarray = None, int_bump_ctrs: np.ndarray = None, int_bump_stds: np.ndarray = None,
                 int_bump_gains: np.ndarray = None):

        """ Creates a new BumpInputWithRecursiveDynamicsScenario object.

        Args:
            n_subjects: The number of subject models to generate.

            n_modes: The number of modes each model should have.  There will be 2*n_modes p and u modes per subject.

            n_neuron_range: The number of neurons each subject has will be pulled uniformly from this range.

            prior_std: The standard deviation of each prior conditional distribution.

            bump_gain_range: We use a Gaussian kernel for the bump functions which form the conditional means for each
            mode. Each bump has a randomly chosen peak value magnitude.  These peak values are pulled uniformly from the
            interval specified by bump_gain_range. See pos_neg_modes below about how the signs of these peaks are
            chosen.

            bump_std_range: The Gaussian kernel functions for the conditional means will be axis aligned with a certain
            standard deviation along each axis.  The standard deviation values along each axis will be pulled uniformly
            from this range.

            noise_range: Range of values to pull psi values from when generating latent regression models

            n_dims: The number of spatial dimensions neurons are arranged in.

            pos_neg_modes: If true, the sign of the peak of modes can be either positive or negative (with 50%
            probability)

            input_bump_ctrs: If not None, the parameters of the bump function forming the means for the input mode
            couplings (and internal p couplings) will not be randomly generated but will instead be specified by the
            user.  input_bump_ctrs[:, i] gives the center for the bump function for the i^th mode.

            input_bump_stds: If not randomly generating parameters for the input mean functions, input_bump_stds[:, i]
            gives the dimension standard deviations for the i^th mode.

            input_bump_gains: If not randomly generating parameters for the input mean functions, input_bump_gains[i]
            gives the gain for the i^th mode.

            int_bump_ctrs, int_bump_stds, int_bump_gains: Analagous to the same for the input modes but specifying the
            parameters of the mean functions for the internal u modes.

        """

        use_fixed_params = [input_bump_ctrs is not None, int_bump_ctrs is not None]

        # Generate the prior distribution for the input & internal u modes
        bump_gain_int_width = bump_max_range[1] - bump_max_range[0]
        bump_std_int_width = bump_std_range[1] - bump_std_range[0]

        for inp_int_i in range(2):
            u_modes = [None]*n_modes
            for m_i in range(n_modes):

                # Create the conditional mean bump function
                if not use_fixed_params[inp_int_i]:
                    gain = np.random.rand(1)*bump_gain_int_width + bump_max_range[0]
                    if pos_neg_modes:
                        sign  = 2*np.random.binomial(1, .5) - 1.0
                    else:
                        sign = 1.0
                    gain = sign*gain
                    dim_stds = torch.Tensor(np.random.rand(n_dims)*bump_std_int_width + bump_std_range[0])
                    ctr = torch.Tensor(np.random.rand(n_dims))
                else:
                    if inp_int_i == 0:
                        gain = input_bump_gains[m_i]
                        ctr = torch.Tensor(input_bump_ctrs[:, m_i])
                        dim_stds = torch.Tensor(input_bump_stds[:, m_i])
                    else:
                        gain = int_bump_gains[m_i]
                        ctr = torch.Tensor(int_bump_ctrs[:, m_i])
                        dim_stds = torch.Tensor(int_bump_stds[:, m_i])

                mn_f = GaussianBumpFcn(ctr=ctr, std=dim_stds, peak_vl=gain)

                # Create the standard deviation function
                std_f = ConstantFcn(c=torch.Tensor([prior_std]))

                u_modes[m_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

            if inp_int_i == 0:
                input_u_prior = CondMatrixProductDistribution(dists=u_modes)
            else:
                internal_u_prior = CondMatrixProductDistribution(dists=u_modes)

        # Set the prior distribution for the internal p modes equal to that for the input u modes.  This ensures
        # that the p modes for internal dynamics will be roughly aligned to the u modes for the input.
        internal_p_prior = input_u_prior

        # Generate each subject model
        subject_mdls = [None]*n_subjects
        for s_i in range(n_subjects):

            n_neurons = np.random.randint(n_neuron_range[0], n_neuron_range[1] + 1)
            neuron_pos = torch.rand([n_neurons, n_dims])

            s = [IdentityS() for n in range(n_neurons)]
            m = ConcatenateMap(np.ones([1, 2], dtype=bool))
            mdl = LatentRegModel(d_in=[n_modes, n_neurons], d_out=[n_neurons], d_proj=[n_modes, n_modes],
                                 d_trans=[2*n_modes], m=m, s=s, noise_range=noise_range)

            input_u_modes = input_u_prior.form_standard_sample(input_u_prior.sample(neuron_pos))
            internal_p_modes = internal_p_prior.form_standard_sample(internal_p_prior.sample(neuron_pos))
            internal_u_modes = internal_u_prior.form_standard_sample(internal_u_prior.sample(neuron_pos))

            mdl.p[0].data = torch.eye(n_modes)
            mdl.p[1].data = internal_p_modes
            mdl.u[0].data = torch.cat((input_u_modes, internal_u_modes), dim=1)
            mdl.neuron_pos = neuron_pos

            subject_mdls[s_i] = mdl

        self.n_modes = n_modes
        self.n_dims = n_dims
        self.input_u_prior = input_u_prior
        self.internal_p_prior = internal_p_prior
        self.internal_u_prior = internal_u_prior
        self.subject_mdls = subject_mdls

    def generate_training_subject_mdl(self, s_i: int, assign_p_u: bool = True) -> LatentRegModel:
        """ Generates a new subject model with random parameters which can be fit to
            data generated from the scenario.

            Args:
                s_i: The subject to generate a model for

                assign_p_u: True if p and u tensors should be created for the model.  Setting this to false
                saves memory if the model will be fit with priors over the modes (in which case the p and u
                modes in the subject model object are ignored.)

            Returns:
                mdl: The generated subject model

        """
        true_mdl = self.subject_mdls[s_i]
        n_neurons = true_mdl.neuron_pos.shape[0]
        s = [IdentityS() for n in range(n_neurons)]
        m = ConcatenateMap(np.ones([1, 2], dtype=bool))
        mdl = LatentRegModel(d_in=[self.n_modes, n_neurons], d_out=[n_neurons],
                                        d_proj=[self.n_modes, self.n_modes],
                                        d_trans=[2*self.n_modes], m=m, s=s,
                                        assign_p_u=assign_p_u)
        if assign_p_u:
            mdl.p[0].data = torch.eye(self.n_modes)
            mdl.p_trainable[0] = False

        return mdl

    def generate_training_collection(self, s_i: int, data: TimeSeriesDataset,
                                     mn_init_mn:float = 0, mn_init_std:float = .01,
                                     std_init_vl:float = .01) -> SubjectVICollection:
        """ Generates a SubjectVICollection for fitting data to a given subject in the scenario.

        Args:
            s_i: The index of the subject to fit to.

            data: The training data for the subject, as a TimeSeriesDataset in the same convention as returned by
            generate_data()

            mn_init_mn, mn_init_std, std_init_vl: Values for initializing distributions on modes.  See
            generate_training_subject_posteriors for more information.

        Returns:
            collection: The generated collection.
        """

        post_dists = self.generate_training_subject_posteriors(s_i=s_i, mn_init_mn=mn_init_mn, mn_init_std=mn_init_std,
                                                        std_init_vl=std_init_vl)

        return SubjectVICollection(s_mdl=self.generate_training_subject_mdl(s_i, assign_p_u=False),
                                   data=data,
                                   p_dists=post_dists[0],
                                   u_dists=post_dists[1],
                                   input_grps=[0, 1],
                                   output_grps=[1],
                                   props=[self.subject_mdls[s_i].neuron_pos],
                                   input_props=[None, 0],
                                   output_props=[0],
                                   min_var=[.00001])

    def generate_training_subject_posteriors(self, s_i: int, mn_init_mn:float = 0, mn_init_std:float = .01,
                                             std_init_vl:float = .01) -> Sequence:
        """ Generates the posterior distributions for a given subject for variational inference.

        Because the input p modes are assumed fixed and known, a tensor is returned in place of a distribution
        for these modes.

        Args:
            s_i: The subject to generate posteriors for

            mn_init_mn, mn_init_std: The mean and standard deviation of the normal distribution the initial mean values
            for each mode distribution are sampled from.

            std_init_vl: The initial standard deviation for each mode distribution.  The standard deviation of each
            conditional distribution is initially the same for all neuron positions.

        Returns:
            dists: dists[0] are the posterior distributions for the p modes.  dists[0][0] is a tensor for the input
            p modes and dists[0][1] is a conditional distribution for the internal p modes. dists[1][0] is a
            conditional distribution for the u modes.
        """

        n_neurons = self.subject_mdls[s_i].neuron_pos.shape[0]

        # The input p tensor is fixed
        input_p = torch.eye(self.n_modes)

        # Generate the distribution over internal p modes
        internal_p_mode_dists = [None]*self.n_modes
        for m_i in range(self.n_modes):
            mn_f = IndSmpConstantRealFcn(n=n_neurons, init_mn=mn_init_mn, init_std=mn_init_std)
            std_f = IndSmpConstantBoundedFcn(n=n_neurons, lower_bound=.00001, upper_bound=10,
                                             init_value=std_init_vl)
            internal_p_mode_dists[m_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        internal_p_dist = CondMatrixProductDistribution(dists=internal_p_mode_dists)

        u_mode_dists = [None]*(2*self.n_modes)
        for m_i in range(2*self.n_modes):
            mn_f = IndSmpConstantRealFcn(n=n_neurons, init_mn=mn_init_mn, init_std=mn_init_std)
            std_f = IndSmpConstantBoundedFcn(n=n_neurons, lower_bound=.00001, upper_bound=10,
                                             init_value=std_init_vl)
            u_mode_dists[m_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        u_dist = CondMatrixProductDistribution(dists=u_mode_dists)

        return [[input_p, internal_p_dist], [u_dist]]

    def generate_fitting_priors(self, n_divisions_per_dim: int = 50, n_div_per_hc_side_per_dim: int = 3,
                               init_std: float = .001, min_std = .00001) -> Sequence:
        """
        Generates priors for fitting multiple models with variational inference.

        Note: Because the input p modes are assumed fixed and known, a prior is not generate for these
        modes and the value None is returned in its place (see below).

        Args:
            n_divisions_per_dim: The number of divisons per dimension to use when generating hypercube functions.  There
            will be the same number of divisions for each dimension

            n_div_per_hc_side_per_dim: The number of divisions per hypercube per dimension to use when generating
            hypercube functions.  This will be the same for all dimensions.

            init_std: The initial standard deviation for the conditional distribution for each mode.  The standard
            deviation will take on this value everywhere.

            min_std: The minimum standard deviation the prior distributions can take on.

        Returns:
            dists: dists[0] is the distributions for the p modes.  dists[0][0] is None to indicate there is no
            prior distribution over the input p modes and dists[0][1] is the distribution over the internal p modes.
            dists[1][0] is the distribution over the u modes.

        Raises:
            ValueError: If min_std or init_std is less than 0
            ValueError: If init_std is less than min_std
       """
        if init_std < 0:
            raise(ValueError('init_std must be greater than 0'))
        if min_std < 0:
            raise(ValueError('min_std must be greater than 0'))
        if init_std < min_std:
            raise(ValueError('min_std must be less than init_std'))

        n_div_per_dim = [n_divisions_per_dim]*self.n_dims
        dim_ranges = np.zeros([self.n_dims, 2])
        dim_ranges[:, 1] = 1.0
        n_div_per_hc = [n_div_per_hc_side_per_dim]*self.n_dims

        # Generate internal p mode distributions
        internal_p_mode_dists = [None]*self.n_modes
        for m_i in range(self.n_modes):
            mn_f = SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=n_div_per_dim, dim_ranges=dim_ranges,
                                                n_div_per_hc_side_per_dim=n_div_per_hc)
            std_f = torch.nn.Sequential(SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=n_div_per_dim, dim_ranges=dim_ranges,
                                                n_div_per_hc_side_per_dim=n_div_per_hc), FixedOffsetExp(o=min_std))
            # Set the initial standard deviation
            std_f[0].b_m.data[:] = np.log(init_std - min_std)/(n_div_per_hc_side_per_dim**self.n_dims)

            internal_p_mode_dists[m_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        internal_p_dist = CondMatrixProductDistribution(dists=internal_p_mode_dists)

        # Generate the u mode distributions
        u_mode_dists = [None]*(2*self.n_modes)
        for m_i in range(2*self.n_modes):
            mn_f = SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=n_div_per_dim, dim_ranges=dim_ranges,
                                                n_div_per_hc_side_per_dim=n_div_per_hc)
            std_f = torch.nn.Sequential(SumOfTiledHyperCubeBasisFcns(n_divisions_per_dim=n_div_per_dim, dim_ranges=dim_ranges,
                                                n_div_per_hc_side_per_dim=n_div_per_hc), FixedOffsetExp(o=min_std))
            # Set the initial standard deviation
            std_f[0].b_m.data[:] = np.log(init_std - min_std)/(n_div_per_hc_side_per_dim**self.n_dims)

            u_mode_dists[m_i] = CondGaussianDistribution(mn_f=mn_f, std_f=std_f)

        u_dist = CondMatrixProductDistribution(dists=u_mode_dists)

        p_dists = [None, internal_p_dist]
        u_dists = [u_dist]

        return [p_dists, u_dists]

    def generate_data(self, stimuli: Sequence[torch.Tensor]) -> Sequence[Sequence]:
        """
        Generates data from subject models given time series of input to each model.

        Args:
            stimuli: stimuli[i] contains the input tensor for subject model i.

        Returns:
            data: data[i] contains data for subject i as a TimeSeriesDataset. The first tensor in
            each dataset will be input stimulus and the second will be neural data.  Data are
            "time locked" that is the t^th point in the returned stimulus and the t^th point in
            the returned neural data is the data that generated the (t+1)^th data point in the neural
            data.  This requires discarding the first point in the provided stimulus, so there will
            only by T-1 data points inthe returned dataset.
        """

        n_subjects = len(stimuli)
        neural_inputs = [None]*n_subjects
        for s_i in range(n_subjects):
            n_smps = stimuli[s_i].shape[0]
            n_neurons = self.subject_mdls[s_i].u[0].shape[0]
            neural_inputs[s_i] = torch.zeros([n_smps, n_neurons])
            neural_inputs[s_i][:] = np.nan
            neural_inputs[s_i][0,:] = 0

        sim_inputs = zip(stimuli, neural_inputs)

        output = [s_m.recursive_generate(x=s_i, r_map=[(0, 1)])
                  for s_m, s_i in zip(self.subject_mdls, sim_inputs)]

        # Now produce a time locked version of stimulus and output
        tl_stimuli = [s_i[1:, :] for s_i in stimuli]
        tl_output = [o_i[0][0:-1,:] for o_i in output]

        return [TimeSeriesDataset([tl_stimuli[s_i], tl_output[s_i]]) for s_i in range(n_subjects)]

    def generate_random_input_data(self, n_smps_per_subject: Sequence[int], input_std: float = 1.0) -> Sequence[Sequence]:
        """ Generates data from each subject model as they receive random input.

        Data for each input is generated iid from a N(0, input_std^2) distribution.

        Args:
            n_smps_per_subject: n_smps_per_subject[i] is the number of random input samples to generate from subject
            i.

            input_std: The standard deviation of the input signals.

        Returns:
           data: data[i] is the data for subject i, the formatted described in generate_data()
        """

        input = [input_std*torch.randn([n_smps, self.n_modes]) for n_smps in n_smps_per_subject]
        return self.generate_data(stimuli=input)

    def generate_one_input_mode_random_input_data(self, n_smps_per_subject: Sequence[int],
                                               input_std: float = 1.0) -> Sequence[Sequence]:
        """ Generates data where only one input mode is excited with random input per subject.

        Args:
            n_smps_per_subject: n_smps_per_subject[i] is the number of random input samples to generate from subject
            i.

            input_std: The standard deviation of the input signals.

        Returns:
           data: data[i] is the data for subject i, the formatted described in generate_data()
        """

        n_subjects = len(n_smps_per_subject)

        input = [None]*n_subjects
        for s_i, n_smps in enumerate(n_smps_per_subject):
            mode_i = s_i % self.n_modes
            s_input = torch.zeros([n_smps, self.n_modes])
            s_input[:, mode_i] = input_std*torch.randn([n_smps])
            input[s_i] = s_input

        return self.generate_data(stimuli=input)


def plot_single_2d_conditional_prior(priors: CondMatrixProductDistribution, mode: int,
                                     dim_sampling: Sequence[Sequence] = [[0, 1, .01], [0, 1, .01]],
                                     range: float = None, plot_mn: bool = True, t: np.ndarray = None):
    """ Plots a prior distribution of a neuron's loading given neuron's location in a 2-d space for a single mode.

    Args:
        priors: The conditional prior distribution over modes.

        mode: The index of the mode for the prior to plot.

        dim_sampling: Each entry of dim_sampling specifies how to sample a dimension in the
        domain of the function.  Each entry is of the form [start, stop, int] where start and
        and stop are the start and stop of the range of values to sample from and int
        is the interval values are sampled from.

        range: If provided, colors for values will saturate at +/- range.  If not provided, this
        is set based on the values to be plotted.

        plot_mn: If true, the mean will be plotted.  If false, the standard deviation is plotted.
    """

    # Determine coordinates we will sample from along each dimension
    coords = [np.arange(ds[0], ds[1], ds[2]) for ds in dim_sampling]

    # Form coordinates of each point we will sample from in a single numpy array
    grid = np.meshgrid(*coords, indexing='ij')
    n_pts = grid[0].size
    flat_grid = torch.Tensor(np.concatenate([g.reshape([n_pts,1]) for g in grid], axis=1))
    n_coords_per_dim = [len(c) for c in coords]

    # Get the mean and standard deviation of the unrotated modes at each point in the grid
    mode_mn = priors(flat_grid).cpu().detach().numpy()
    mode_std = torch.cat([d(flat_grid) for d in priors.dists], dim=1).cpu().detach().numpy()

    # Rotate modes
    if t is None:
        n_modes = mode_mn.shape[1]
        t = np.eye(n_modes)

    mode_mn = np.matmul(mode_mn, t)
    mode_std = np.sqrt(np.matmul(mode_std**2, t**2))

    # Get the mode we are to plot
    mode_mn = mode_mn[:, mode]
    mode_std = mode_std[:, mode]

    # Put the mean and standard deviation in images
    mode_mn = mode_mn.reshape(n_coords_per_dim)
    mode_std = mode_std.reshape(n_coords_per_dim)

    # Determine the axes ratio of the images we will plot
    aspect_ratio = float(dim_sampling[0][2])/float(dim_sampling[1][2])

    # Get the mean or std image
    if plot_mn:
        img = mode_mn.transpose()
    else:
        img = mode_std.transpose()

    # Get the max absolute values for the images
    if range is None:
        range = np.max(np.abs(img))

    if plot_mn:
        vmin = -range
        cmap = 'PiYG'
    else:
        vmin = 0
        cmap = 'cool'
    vmax = range

    # Plot image
    plot_axes = plt.axes()
    plotted_im = plot_axes.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plot_axes.xaxis.set_visible(False)
    plot_axes.yaxis.set_visible(False)
    plt.colorbar(mappable=plotted_im, ax=plot_axes)


def plot_2d_conditional_prior(priors: CondMatrixProductDistribution,
                              dim_sampling: Sequence[Sequence] = [[0, 1, .01], [0, 1, .01]],
                              mn_range: float = None, std_range: float = None):
    """ Plots a conditional prior distribution of a neuron's loading given neurons location in a 2-d space.

    The spatial means and standard deviations of each mode will be plotted in a 2 by M grid layout - each column
    is a mode with means plotted on top and standard deviations plotted on bottom.

    Args:
        priors: The conditional prior distribution over modes.

        dim_sampling: Each entry of dim_sampling specifies how to sample a dimension in the
        domain of the fuction.  Each entry is of the form [start, stop, int] where start and
        and stop are the start and stop of the range of values to sample from and int
        is the interval values are sampled from.

        mn_range: If provided, colors for mean values will saturate at +/- mn_range.  If not provided, this
        is set based on the means to be plotted.

        std_range: If provided, colors for standard deviation values will saturate at std_range.  If not provided,
        this is set based on the standard deviation values to be plotted.
    """

    # Generate axes to plot into
    n_modes = len(priors.dists)
    plot_axes = [plt.subplot(2, n_modes, p + 1) for p in range(2*n_modes)]
    mn_axes = [plot_axes[i] for i in range(0, n_modes, 1)]
    std_axes = [plot_axes[i] for i in range(n_modes, 2*n_modes, 1)]

    # Determine the axes ratio of the images we will plot
    aspect_ratio = float(dim_sampling[0][2])/float(dim_sampling[1][2])

    # Get the mean and standard deviation images
    mn_imgs = [generate_image_from_fcn(torch_mod_to_fcn(d.mn_f), dim_sampling)[0].transpose() for d in priors.dists]
    std_imgs = [generate_image_from_fcn(torch_mod_to_fcn(d.std_f), dim_sampling)[0].transpose() for d in priors.dists]

    # Get the max absolute values for the images
    if mn_range is None:
        mn_range = np.max([np.max(np.abs(im)) for im in mn_imgs])
    if std_range is None:
        std_range = np.max([np.max(im) for im in std_imgs])

    # Plot means
    for m_i in range(n_modes):
        mn_im = mn_axes[m_i].imshow(X=mn_imgs[m_i], cmap='PiYG', vmin=-mn_range, vmax=mn_range, origin='lower')
        mn_axes[m_i].set_aspect(aspect_ratio)
        mn_axes[m_i].xaxis.set_visible(False)
        mn_axes[m_i].yaxis.set_visible(False)
        if m_i == n_modes - 1:
            plt.colorbar(mappable=mn_im, ax=mn_axes[m_i])

    # Plot standard deviations
    for m_i in range(n_modes):
        std_im = std_axes[m_i].imshow(std_imgs[m_i], cmap='cool', vmin=0, vmax=std_range, origin='lower')
        std_axes[m_i].xaxis.set_visible(False)
        std_axes[m_i].yaxis.set_visible(False)
        if m_i == n_modes - 1:
            plt.colorbar(mappable=std_im, ax=std_axes[m_i])


def plot_single_2d_mode(mode: np.ndarray, neuron_p: np.ndarray, vl_range: float = None,
                                  dim_range: Sequence[Sequence] = [[0, 1], [0, 1]],
                                  plot_axes: matplotlib.axes = None, plot_color_bar: bool = True):
    """ Plots the mode loadings for a single mode for neurons positioned in 2-d space.

    Args:
        mode: The mode to plot as a 1-d array of length n_neurons.

        neuron_p: The position of each neuron.  Shape is n_neurons*2.

        vl_range: The min and max value to clip values at when assigning colors.  If not provided, will be
        assigned based on the values in modes.

        dim_range: The spatial range to generate plots for.

        plot_axes: The axes to plot into.  If none, one will be provided

        plot_color_bar: True if a color bar should be added to the plot.
    """

    if vl_range is None:
        vl_range = np.max(np.abs(mode))

    if plot_axes is None:
        plot_axes = plt.axes()

    # Plot the mode
    sp = plot_axes.scatter(x=neuron_p[:,0], y=neuron_p[:,1], marker='o', c=mode,
                               cmap='PiYG', vmin=-vl_range, vmax=vl_range)

    plot_axes.set_xlim(dim_range[0])
    plot_axes.set_ylim(dim_range[1])
    plot_axes.set_aspect('equal')

    plot_axes.xaxis.set_visible(False)
    plot_axes.yaxis.set_visible(False)

    if plot_color_bar:
        cb = plt.colorbar(mappable=sp)


def plot_single_2d_mode_posterior(post: CondMatrixProductDistribution, mode: int, neuron_p: np.ndarray,
                                  vl_range: float = None, dim_range: Sequence[Sequence] = [[0, 1], [0, 1]],
                                  plot_axes: matplotlib.axes = None, plot_mn: bool = True, t: np.ndarray = None,
                                  plot_color_bar: bool = True):
    """ Plots the mean or standard deviation of mode loadings for a single mode for neurons positioned in 2-d space.

    The distribution can be transformed before plotting.

    Args:
        post: The posterior distribution over modes.

        mode: The index of the mode to plot the posterior over.

        neuron_p: The position of each neuron.  Shape is n_neurons*2.

        vl_range: The min and max value to clip values at when assigning colors.  If not provided, will be
        assigned based on the values in modes.

        dim_range: The spatial range to generate plots for.

        plot_axes: Axes to plot into.  If None, one will be created.

        plot_mn: If true, the mean is plotted.  If false, the standard deviation is plotted.

        t: The transition matrix to apply to the distribution if it should be transformed before plotting.

        plot_color_bar: True if a color bar should be added to the plot.

    """

    # Get means and standard deviations for unrotated modes
    mode_mn = post(neuron_p).cpu().detach().numpy()
    mode_std = torch.cat([d(neuron_p) for d in post.dists], dim=1).cpu().detach().numpy()

    # Rotate modes
    if t is None:
        n_modes = mode_mn.shape[1]
        t = np.eye(n_modes)

    mode_mn = np.matmul(mode_mn, t)
    mode_std = np.sqrt(np.matmul(mode_std**2, t**2))

    # Get the mode we are to plot
    mode_mn = mode_mn[:, mode]
    mode_std = mode_std[:, mode]

    # Plot either the mean or standard deviation
    if plot_mn:
        mode_im = mode_mn
    else:
        mode_im = mode_std

    # Get the max absolute values for the images
    if vl_range is None:
        vl_range = np.max(np.abs(mode_im))

    if plot_mn:
        vmin = -vl_range
        cmap = 'PiYG'
    else:
        vmin = 0
        cmap = 'cool'
    vmax = vl_range

    # Plot the mode
    if plot_axes is None:
        plot_axes = plt.axes()
    sp = plot_axes.scatter(x=neuron_p[:,0], y=neuron_p[:,1], marker='o', c=mode_im,
                               cmap=cmap, vmin=vmin, vmax=vmax)

    plot_axes.set_xlim(dim_range[0])
    plot_axes.set_ylim(dim_range[1])
    plot_axes.set_aspect('equal')

    plot_axes.xaxis.set_visible(False)
    plot_axes.yaxis.set_visible(False)

    if plot_color_bar:
        cb = plt.colorbar(mappable=sp)


def plot_2d_modes(modes: np.ndarray, neuron_p: np.ndarray, vl_range: float = None,
                  dim_range: Sequence[Sequence] = [[0, 1], [0, 1]]):
    """ Plots the mode loadings for neurons positioned in a 2-d space.

    Modes will be plotted in a row, followed by a colorbar.  All modes will be plotted on the same color scale.

    Args:
        modes: The modes to plot as a n_neurons by n_modes array. Each column is a mode.

        neuron_p: The position of each neuron.  Shape is n_neurons*2.

        vl_range: The min and max value to clip values at when assigning colors.  If not provided, will be
        assigned based on the values in modes.

        dim_range: The spatial range to generate plots for.
    """

    n_neurons, n_modes = modes.shape

    if vl_range is None:
        vl_range = np.max(np.abs(modes))

    # Plot images for each mode
    mode_axes = [plt.subplot(1, n_modes+1, m_i+1) for m_i in range(n_modes+1)]
    for m_i in range(n_modes):
        sp = mode_axes[m_i].scatter(x=neuron_p[:,0], y=neuron_p[:,1], marker='o', c=modes[:, m_i],
                               cmap='PiYG', vmin=-vl_range, vmax=vl_range)
        mode_axes[m_i].set_xlim(dim_range[0])
        mode_axes[m_i].set_ylim(dim_range[1])
        mode_axes[m_i].set_aspect('equal')

        mode_axes[m_i].xaxis.set_visible(False)
        mode_axes[m_i].yaxis.set_visible(False)

        if m_i == n_modes - 1:
            # Plot colorbar
            cb = plt.colorbar(mappable=sp, cax=mode_axes[m_i+1])

            # Make sure the colorbar axes are shaped as we want them
            data_points = mode_axes[m_i].get_position().get_points()
            cb_points = mode_axes[m_i+1].get_position().get_points()
            cb_height = mode_axes[m_i].get_position().height

            x0 = cb_points[0][0]
            y0 = data_points[0][1]

            new_cb_points = np.asarray([[x0, y0],
                                        [x0 + cb_height*.05, y0 + cb_height]])
            new_cb_pos = matplotlib.transforms.Bbox(points=new_cb_points)
            mode_axes[m_i+1].set_position(new_cb_pos)


def learn_prior_transformation(d0, d1, dim_sampling: Sequence[Sequence] = [[0, 1, .01], [0, 1, .01]],
                               d0_slice: slice = None, d1_slice: slice = None):
    """ Learns a transformation between two the means of two prior distributions.

    The transformation is learned based on sampling both conditional distributions.

    Args:
        d0: The distribution to transform *to*

        d1: The distribution to transform *from*

        dim_sampling: Each entry of dim_sampling specifies how to sample the conditional prior.  Each entry is
        of the form [start, stop, int] where start and and stop are the start and stop of the range of values to
        sample from and int is the interval values are sampled from.

        d0_slice: A slice indicating which modes of d0 to learn the transformation to.

        d1_slice: A slice indicating which modes of d1 to learn the transformation from.

    """

    # Determine coordinates we will sample from along each dimension
    coords = [np.arange(ds[0], ds[1], ds[2]) for ds in dim_sampling]

    # Form coordinates of each point we will sample from in a single numpy array
    grid = np.meshgrid(*coords, indexing='ij')
    n_pts = grid[0].size
    flat_grid = torch.Tensor(np.concatenate([g.reshape([n_pts,1]) for g in grid], axis=1))

    mn_0 = d0(flat_grid).cpu().detach().numpy()
    if d0_slice is not None:
        mn_0 = mn_0[:, d0_slice]

    mn_1 = d1(flat_grid).cpu().detach().numpy()
    if d1_slice is not None:
        mn_1 = mn_1[:, d1_slice]

    solution = np.linalg.lstsq(mn_1, mn_0, rcond=None)
    return solution[0]


