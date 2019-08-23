""" Tools for fitting latent regression models with variational inference. """

import itertools
import time
from typing import Sequence, Union


import numpy as np
import torch

from janelia_core.ml.datasets import TimeSeriesDataset
from janelia_core.ml.datasets import TimeSeriesBatch
from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.torch_distributions import CondVAEDistriubtion
from janelia_core.ml.torch_distributions import CondMatrixProductDistribution

from janelia_core.ml.datasets import cat_time_series_batches
from janelia_core.ml.utils import format_and_check_learning_rates


def format_output_list(base_str: str, it_str: str, vls: Sequence[float], inds: Sequence[int]):
    """ Produces a string to display a list of outputs.

    String will be of the format:

        base_str + ' ' + it_str+ints[0] + ': ' + str(vls[0]) + ', ' + it_str+inds[1] + ': ' + str(vls[1]) + ...

    Args:
        base_str: The base string that preceeds everything else on the string

        it_str: The string that should come befor each printed value

        vls: The values that should be printed

        inds: The indices to associate with each printed value

    Returns:
        f_str: The formatted string
    """

    n_vls = len(vls)

    f_str = base_str + ' '

    for i, vl in enumerate(vls):
        f_str += it_str + str(inds[i]) + ': {' + str(i) + ':.2e}'
        if i < n_vls - 1:
            f_str += ', '

    # Format values
    f_str = f_str.format(*vls)

    return f_str


def vae_fit_latent_reg_model(l_mdl: LatentRegModel, q_p_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             q_u_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             prior_p_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             prior_u_dists: Sequence[Sequence[CondVAEDistriubtion]],
                             x: Sequence[torch.Tensor], y: Sequence[torch.Tensor], x_props: Sequence[torch.Tensor],
                             y_props: Sequence[torch.Tensor], batch_size: int=100, send_size: int=100, max_its: int=10,
                             learning_rates=.01, adam_params: dict = {}, min_var: float=0.0, update_int: int=100,
                             fit_priors: bool = True, grp_w: torch.Tensor = None):

    """ A function for fitting a latent regression model and a prior over it's modes with variational inference.

    Note: When calling this function the values of the p and u parameters of the latent regression model are
    ignored (since these represent point values and this function fits a distribution over the modes).

    Note: This function will move batches of data to whatever device the latent regression and distribution parameters
    are on.  (We implicitly assume the latent regression model and all mode distributions are on the same device.)
    Property data will also be moved to this device.

    Args:

        l_mdl: The latent regression model to fit.

        q_p_dists, q_u_dists: Inference distributions for each mode.  q_p_dists[g][j] is the distribution over the j^th
        column of the p_g matrix of a LatentRegModel. q_u_dists[h][j] is the same for the j^th column of the u_h matrix.

        prior_p_dists, prior_u_dists: Prior distributions for each mode.  prior_p_dists[g][j] is the distribution over
        the j^th column of p_g, and prior_u_dists[h][j] is the same for the j^th column of u_h.

        x: A sequence of inputs.  x[g] contains the input tensor for group g.  x[g] should be of
            shape n_smps*d_in[g]

        y: A sequence of outputs.  y[h] contains the output tensor for group h.  y[h] should be of
        shape n_smps*d_out[h]

        x_props: A sequence of properties for variables in x.  (E.g., if x is neural activity, this is the
        properties, such as position, for each neuron.)  x_props[g][j,:] contains the properties for variable j in
        group g.

        y_props: A sequence of properties for variables in y. y_props[h][j,:] contains the properties for variable j
        in group h.

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

        update_int: The interval that updates should be printed

        fit_priors: If false, fitting is done where the only term that is optimized is negative log-likelihood (KL
        divergence between priors and q is omitted).  Setting this to false, may be helpful for doing an initial run to
        fit model parameters.

        grp_w: grp_w[i] is the weight to apply for group i when summing log-likelihoods across groups.
        If None, no weighting will be used.

    Returns:
        log: A dictionary logging progress.  Will have the entries:
            'elapsed_time': log['elapsed_time'][i] contains the elapsed time from the beginning of optimization to
            the end of iteration i

            'obj': log['obj'][i] contains the objective value at the beginning (before parameters are updated) of
            iteration i.

    Raises:

        ValueError: If send_size is larger than batch size

    """

    if send_size > batch_size:
        raise(ValueError('send_size must be less than or equal to batch size'))

    # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
    # where the learning rate starting at iteration 0 is guaranteed to be listed first
    learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

    # Get a list of all the parameters we need to fit
    subject_params = l_mdl.vae_parameters()
    subject_params = [p[1] for p in subject_params] # Keep only parameters, discarding names
    q_p_params = itertools.chain(*[d.r_params() for q_p_g_dists in q_p_dists for d in q_p_g_dists])
    q_u_params = itertools.chain(*[d.r_params() for q_u_g_dists in q_u_dists for d in q_u_g_dists])
    prior_p_params = itertools.chain(*[d.r_params() for prior_p_g_dists in prior_p_dists for d in prior_p_g_dists])
    prior_u_params = itertools.chain(*[d.r_params() for prior_u_g_dists in prior_u_dists for d in prior_u_g_dists])
    parameters = list(itertools.chain(subject_params, q_p_params, q_u_params, prior_p_params, prior_u_params))

    # See what device parameters are on
    device = parameters[0].device
    run_on_gpu = device.type == 'cuda'

    # See what memory usage on GPU is before we've moved any data there
    if run_on_gpu:
        init_gpu_mem_usage = torch.cuda.memory_allocated()
        print('Initial GPU memory usage: ' + str(init_gpu_mem_usage) + ' bytes.')

    # Move property data to the device
    x_props = [props.to(device) for props in x_props]
    y_props = [props.to(device) for props in y_props]

    # See what memory usage on GPU is after sending neuron properties
    if run_on_gpu:
        after_props_gpu_mem_usage = torch.cuda.memory_allocated()
        print('GPU memory usage after sending properties: ' + str(after_props_gpu_mem_usage) + ' bytes.')

    # Setup optimizer
    optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0], **adam_params)

    # Calculate the correction factor we apply when calculating negative
    # log-likelihood to account for the fact that our batch sizes don't use
    # all samples - this is to prevent batch_size as effectively acting
    # as a tuning parameter (because if we don't apply this correction, smaller
    # batch sizes will mean the priors have more influence - we want the influence
    # of the priors to be determined only by the total amount of data we fit on
    n_smps = x[0].shape[0]
    batch_ratio = float(n_smps)/batch_size

    # See how many input and output groups we have
    n_input_grps = len(q_p_dists)
    n_output_grps = len(q_u_dists)

    # Setup variables we will need for fitting
    cur_it = 0
    start_time = time.time()
    prev_learning_rate = learning_rate_values[0]

    elapsed_time_log = np.zeros(max_its)
    elbo_log = np.zeros(max_its)

    # Perform fitting
    while cur_it < max_its:

            elapsed_time = time.time() - start_time  # Record elapsed time here because we measure it from the start of
            # each iteration.  This is because we also record the objective value for each iteration before parameters
            # are updated.  In this way, the elapsed time is the elapsed time to get to a set of parameters for which we
            # report the objective value

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= cur_it)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind]
            if cur_learning_rate != prev_learning_rate:
                # We reset the whole optimizer because ADAM is an adaptive optimizer
                optimizer = torch.optim.Adam(parameters, lr=cur_learning_rate, **adam_params)
                prev_learning_rate = cur_learning_rate

            # Chose the data samples for this iteration:
            cur_smps = np.random.choice(n_smps, batch_size, replace=False)
            batch_x = [x_g[cur_smps, :] for x_g in x]
            batch_y = [y_h[cur_smps, :] for y_h in y]

            # Zero the gradients to prepare for this optimization step
            optimizer.zero_grad()

            # Sample from the q distribution
            q_p_smps = [[d.form_standard_sample(d.sample(g_props))for d in q_p_g_dists]
                        for q_p_g_dists, g_props in zip(q_p_dists, x_props)]
            q_p_smps_t = [torch.cat(smps_g, dim=1) for smps_g in q_p_smps]

            q_u_smps = [[d.form_standard_sample(d.sample(g_props)) for d in q_u_g_dists]
                        for q_u_g_dists, g_props in zip(q_u_dists, y_props)]
            q_u_smps_t = [torch.cat(smps_g, dim=1) for smps_g in q_u_smps]

            # Send data to device in small chunks (if send_size < batch_size) to calculate negative log-likelhood
            start_ind = 0
            end_ind = np.min([batch_size, send_size])
            neg_ll = 0
            elbo_db = 0
            while True:
                sent_x = [batch_x_g[start_ind:end_ind, :].to(device) for batch_x_g in batch_x]
                sent_y = [batch_y_h[start_ind:end_ind, :].to(device) for batch_y_h in batch_y]

                sent_y_hat = l_mdl.cond_forward(x=sent_x, p=q_p_smps_t, u=q_u_smps_t)
                sent_nll = batch_ratio*l_mdl.neg_ll(y=sent_y, mn=sent_y_hat, w=grp_w)

                elbo_db += sent_nll
                #sent_nll.backward(retain_graph=True)

                # We call backward on each sent chunk of data but we still need to accumulate our
                # total negative log likelihood term for the elbo
                neg_ll += sent_nll.detach().cpu().numpy()

                if end_ind == batch_size:
                    break

                start_ind = end_ind
                end_ind = np.min([batch_size, start_ind + send_size])

            if fit_priors:
                # Calculate kl divergence between conditional posterior and priors for p modes
                kl_p = [None]*n_input_grps # Keep track of KL divergence for each mode for diagnostic purposes
                for g in range(n_input_grps):
                    q_p_mode_dists = q_p_dists[g]
                    prior_p_mode_dists = prior_p_dists[g]

                    n_p_mode_dists = len(q_p_mode_dists)
                    p_mode_kls = np.zeros(n_p_mode_dists)
                    for m_i in range(n_p_mode_dists):
                        mode_kl = torch.sum(q_p_mode_dists[m_i].kl(d_2=prior_p_mode_dists[m_i], x=x_props[g],
                                                               smp=q_p_smps[g][m_i]))
                        elbo_db += mode_kl
                        #mode_kl.backward()
                        p_mode_kls[m_i] = mode_kl.detach().cpu().numpy()
                    kl_p[g] = p_mode_kls

                # Calculate kl divergence between conditional posterior and priors for u modes
                kl_u = [None]*n_output_grps
                for h in range(n_output_grps):
                    q_u_mode_dists = q_u_dists[h]
                    prior_u_mode_dists = prior_u_dists[h]

                    n_u_mode_dists = len(q_u_mode_dists)
                    u_mode_kls = np.zeros(n_u_mode_dists)
                    for m_i in range(n_u_mode_dists):
                        mode_kl = torch.sum(q_u_mode_dists[m_i].kl(d_2=prior_u_mode_dists[m_i], x=y_props[h],
                                                                smp=q_u_smps[h][m_i]))
                        elbo_db += mode_kl
                        #mode_kl.backward()
                        u_mode_kls[m_i] = mode_kl.detach().cpu().numpy()
                    kl_u[h] = u_mode_kls

            # Take a step here
            elbo_db.backward()
            optimizer.step()

            # Calculate the value of the ELBO here
            if fit_priors:
                kl_p_sum = np.sum([np.sum(kl_p_g) for kl_p_g in kl_p])
                kl_u_sum = np.sum([np.sum(kl_u_g) for kl_u_g in kl_u])
            else:
                kl_p_sum = 0
                kl_u_sum = 0
            neg_elbo = neg_ll + kl_p_sum + kl_u_sum

            # Correct any noise variances that are too small
            with torch.no_grad():
                for psi_h in l_mdl.psi:
                    small_psi_inds = torch.nonzero(psi_h < min_var)
                    psi_h.data[small_psi_inds] = min_var

            # Log our progress
            elapsed_time_log[cur_it] = elapsed_time
            elbo_log[cur_it] = -1*neg_elbo

            # Provide user with some feedback of requested
            if cur_it % update_int == 0:
                if run_on_gpu:
                    cur_gpu_mem_usage = torch.cuda.memory_allocated()
                else:
                    cur_gpu_mem_usage = np.nan
                print('It: ' + str(cur_it) + ': Elapsed fitting time ' + str(elapsed_time) +
                      ', elbo: ' + str(-1*neg_elbo) + ', lr: ' + str(cur_learning_rate) +
                      ', GPU mem. usage: ' + str(cur_gpu_mem_usage) + ' bytes')
                print('    ll: ' + str(-1*neg_ll) + ', kl_p_sum: ' + str(kl_p_sum) + ', kl_u_sum: ' + str(kl_u_sum))

            cur_it += 1

    # Give final fitting results (if we have not already)
    if (cur_it - 1) % update_int != 0:
        print('It: ' + str(cur_it - 1) + ': Elapsed fitting time ' + str(elapsed_time) +
              ', elbo: ' + str(-1*neg_elbo) + ', lr: ' + str(cur_learning_rate) +
              ', GPU mem. usage: ' + str(cur_gpu_mem_usage) + ' bytes')
        print('    ll: ' + str(-1*neg_ll) + ', kl_p_sum: ' + str(kl_p_sum) + ', kl_u_sum: ' + str(kl_u_sum))

    # Format output
    log = {'elapsed_time': elapsed_time_log, 'elbo': elbo_log}

    return log


class SubjectVICollection():
    """ Holds data, likelihood models and posteriors for fitting data to a single subject with variational inference."""

    def __init__(self, s_mdl: LatentRegModel, p_dists: Sequence, u_dists: Sequence,
                 p_ri_fcns: Sequence, u_ri_fcns: Sequence,
                 data: Union[TimeSeriesBatch, TimeSeriesDataset], input_grps: Sequence, output_grps: Sequence,
                 props: Sequence, input_props: Sequence, output_props: Sequence, min_var: Sequence[float]):
        """ Creates a new SubjectVICollection object.

        Args:
            s_mdl: The likelihood model for the subject.

            u_dists: The posterior distributions for the u modes.  u_modes[h] is either:
                1) A janelia_core.ml.torch_distributions import CondVAEDistriubtion
                2) A torch tensor (if there is no distribution for the u modes for group h

            p_dists: The posterior distributions for the p modes.  Same form as u_dists.

            p_ri_fcns: p_ri_fcns[g] contains a function to apply to re-initialize p_dists[g].  If p_dists[g]
            is a tensor this should be none.

            u_ri_fcns: u_ri_fcns[h] contains a function to apply to re-initialize u_dists[h].  If u_dists[h]
            is a tensor this should be none.

            data: Data for the subject.

            input_grps: input_grps[g] is the index into data.data for the g^th input group

            output_grps: output_grps[h] is the index into data.data for the h^th output group

            props: props[i] is a tensor of properties for one or more input or output groups of variables

            input_props: input_props[g] is the index into props for the properties for the g^th input group.  If
            the modes for the g^th input group are fixed, then input_props[g] should be None.

            output_props: output_props[h[ is the index into props for the properties of the g^th output group.  If
            the modes for the h^th output group are fixed, then outpout_props[h] should be None.

            min_var: min_var[h] is the minimum variance for the additive noise variables for output group h

        """

        self.s_mdl = s_mdl
        self.p_dists = p_dists
        self.u_dists = u_dists
        self.p_ri_fcns = p_ri_fcns
        self.u_ri_fcns = u_ri_fcns
        self.data = data
        self.input_grps = input_grps
        self.output_grps = output_grps
        self.props = props
        self.input_props = input_props
        self.output_props = output_props
        self.min_var = min_var
        self.device = None # Initially we don't specify which device everything is on (allowing things to potentially
                           # be on multiple devices).

    def trainable_parameters(self) -> Sequence:
        """ Returns all trainable parameters for the collection.

        Returns:
            params: The list of parameters.
        """

        p_dist_params = itertools.chain(*[d.parameters() for d in self.p_dists if not isinstance(d, torch.Tensor)])
        u_dist_params = itertools.chain(*[d.parameters() for d in self.u_dists if not isinstance(d, torch.Tensor)])

        return list(itertools.chain(p_dist_params, u_dist_params, self.s_mdl.trainable_parameters()))

    def to(self, device: Union[torch.device, int]):
        """ Moves all relevant attributes of the collection to the specified device.

        This will move the subject model, posterior distributions for the p & u modes and variable
        properties to the device.

        Note that fitting data will not be moved to the device (as we expect in general the full set
        of training data will be too large to move to GPU memory).

        Args:
            device: The device to move attributes to.
        """

        self.s_mdl = self.s_mdl.to(device)
        self.p_dists = [d.to(device) for d in self.p_dists]
        self.u_dists = [d.to(device) for d in self.u_dists]
        self.props = [p.to(device) for p in self.props]
        self.device = device


class MultiSubjectVIFitter():
    """ Object for fitting a collection of latent regression models with variational inference.

    """

    def __init__(self, p_priors: Sequence[CondMatrixProductDistribution],
                 u_priors: Sequence[CondMatrixProductDistribution], s_collections: Sequence[SubjectVICollection]):
        """ Creates a new MultiSubjectVIFitter object.

        Args:
            s_collections: A set of SubjectVICollections to use when fitting data.

            p_priors: The conditional priors for the p modes of each input group.  If the modes for a
            group are fixed, the entry in p_priors for that group should be None.

            u_priors: The conditional priors for the u modes for each output group, same format as p_priors.
        """

        self.s_collections = s_collections
        self.p_priors = p_priors
        self.u_priors = u_priors

        # Attributes for keeping track of which devices everything is on
        self.prior_device = None
        self.s_collection_devices = [None]*len(self.s_collections)
        self.distributed = False  # Keep track if we have distributed everything yet

    def distribute(self, devices: Sequence[Union[torch.device, int]], s_inds: Sequence[int] = None):
        """ Distributes prior collections as well as 0 or more subject models across devices.

        Args:
            devices: Devices that priors and subject collections should be distributed across.

            s_inds: Indices into self.s_collections for subject models which should be distributed across devices.
            If none, all subject models will be distributed.

        """
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        n_devices = len(devices)
        n_dist_mdls = len(s_inds)

        # By convention priors go onto first device
        self.p_priors = [d.to(devices[0]) if d is not None else None for d in self.p_priors]
        self.u_priors = [d.to(devices[0]) if d is not None else None for d in self.u_priors]
        self.prior_device = devices[0]

        # Distribute subject collections
        for i in range(n_dist_mdls):
            device_ind = (i+1) % n_devices
            self.s_collections[s_inds[i]].to(devices[device_ind])
            self.s_collection_devices[s_inds[i]] = devices[device_ind]

        self.distributed = True

    def trainable_parameters(self, s_inds: Sequence[int] = None) -> Sequence:
        """ Gets all trainable parameters for fitting priors and a set of subjects.

        Args:
            s_inds: Specifies the indices of subjects that will be fit.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

        Returns:
             params: Requested parameters.
        """

        if s_inds is None:
            s_inds = range(len(self.s_collections))

        p_params = itertools.chain(*[d.parameters() for d in self.p_priors if d is not None])
        u_params = itertools.chain(*[d.parameters() for d in self.u_priors if d is not None])
        collection_params = itertools.chain(*[self.s_collections[s_i].trainable_parameters() for s_i in s_inds])

        return list(itertools.chain(p_params, u_params, collection_params))

    def generate_data_loaders(self, n_batches: int, s_inds: Sequence[int] = None,
                              pin_memory: bool = False) -> Sequence:
        """ Generates data loaders for the data for a given set of subjects.

        This will return data loaders which return random samples in a given number of batches.  If different
        subjects have different numbers of total samples, the number of samples returned per batch for each
        subject will be different so that the total dataset for each subject is cycled through in the specified
        number of batches.

        Args:
            n_batches: The number of batches to break the data up for each subject into.

            s_inds: Specifies the indices of subjects that will be fit.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, s_inds = range(n_subjects).

            pin_memory: True if data loaders should return data in pinned memory

        Returns:
            data_loaders: data_loaders[i] is the data loader for the i^th subject in s_inds.

        Raises:
            ValueError: If n_batches is greater than the number of samples for any requested subject.
        """
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        n_smps = [len(self.s_collections[s_i].data) for s_i in s_inds]

        for i, n_s in enumerate(n_smps):
            if n_s < n_batches:
                raise(ValueError('Subject ' + str(s_inds[i]) + ' has only ' + str(n_s) + ' samples, while '
                      + str(n_batches) + ' batches requested.'))

        batch_sizes = [int(np.ceil(float(n_s)/n_batches)) for n_s in n_smps]
        return [torch.utils.data.DataLoader(dataset=self.s_collections[s_i].data, batch_size=batch_sizes[i],
                                            collate_fn=cat_time_series_batches, pin_memory=pin_memory,
                                            shuffle=True)
                for i, s_i in enumerate(s_inds)]

    def get_device_memory_usage(self) -> Sequence:
        """ Returns the memory usage of each device in the fitter.

        The memory usage for cpu devices will be nan.

        Returns:
            m_usage: m_usage[i] is the amount of memory used on the i^th device
        """

        all_devices = set(self.s_collection_devices + [self.prior_device])
        return [torch.cuda.memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in all_devices]

    def get_device_max_memory_allocated(self) -> Sequence:
        """ Returns the max memory usage of each device in the fitter.

        The memory usage for cpu devices will be nan.

        Returns:
            m_usage: m_usage[i] is the max amount of memory used on the i^th device
        """

        all_devices = set(self.s_collection_devices + [self.prior_device])
        return [torch.cuda.max_memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in all_devices]

    def fit(self, n_epochs: int = 10, n_batches: int = 10, learning_rates = .01,
            adam_params: dict = {}, s_inds: Sequence[int] = None, pin_memory: bool = False,
            update_int: int = 1, print_mdl_nlls: bool = True, print_sub_kls: bool = True,
            print_memory_usage: bool = True, post_reinit_int: int = 10):
        """

        Args:

            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the training data up into per epoch.  When multiple subjects have
            different numbers of total training samples, the batch size for each subject will be selected so we go
            through the entire training set for each subject after processing n_batches each epoch.

            learning_rates: If a single number, this is the learning rate to use for all epochs.  Alternatively, this
            can be a list of tuples.  Each tuple is of the form (epoch, learning_rate), which gives the learning rate
            to use from that epoch onwards, until another tuple specifies another learning rate to use at a different
            epoch on.  E.g., learning_rates = [(0, .01), (1000, .001), (10000, .0001)] would specify a learning
            rate of .01 from epoch 0 to 999, .001 from epoch 1000 to 9999 and .0001 from epoch 10000 onwards.

            adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object.
            Note that if learning rate is specified here *it will be ignored.* (Use the learning_rates option instead).

            s_inds: Specifies the indices of subjects to fit to.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

            pin_memory: True if data for training should be copied to CUDA pinned memory

            update_int: Fitting status will be printed to screen every update_int number of epochs

            print_mdl_nlls: If true, when fitting status is printed to screen, the negative log likelihood of each
            evaluated model will be printed to screen.

            print_sub_kls: If true, when fitting status is printed to screen, the kl divergence for the p and u modes
            for each fit subject will be printed to screen.

            print_memory_usage: If true, when fitting status is printed to screen, the memory usage of each
            device will be printed to streen.

            post_reinit_int: The interval at which posterior distributions should be re-initialized.  The posterior
            distributions for one subject will be re-initialized every this many epochs.  Subjects are cycled
            through for re-initialization.

        Return:
            log: A dictionary with the following entries:

                'elapsed_time': A numpy array of the elapsed time for completion of each epoch

                'mdl_nll': mdl_nll[e, i] is the negative log likelihood for the subject model s_inds[i] at the start
                of epoch e (that is when the objective has been calculated but before parameters have been updated)

                'sub_p_kl': sub_p_kl[e,i] is the kl divergence between the posterior and conditional prior for the p
                modes for subject i at the start of epoch e.

                'sub_u_kl': sub_u_kl[e,i] is the kl divergence between the posterior and conditional prior the u modes
                for subject i at the start of epoch e.

                nelbo: nelbo[e] contains the negative elbo at the start of epoch e

        Raises:
            RuntimeError: If distribute() has not been called before fitting.

        """

        if not self.distributed:
            raise(RuntimeError('self.distribute() must be called before fitting.'))

        t_start = time.time()  # Get starting time

        # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
        # where the learning rate starting at iteration 0 is guaranteed to be listed first
        learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

        # Determine what subjects we are fitting for
        if s_inds is None:
            n_subjects = len(self.s_collections)
            s_inds = range(n_subjects)
        n_fit_subjects = len(s_inds)

        # Get data loaders for the subjects
        data_loaders = self.generate_data_loaders(n_batches=n_batches, s_inds=s_inds, pin_memory=pin_memory)
        n_smp_data_points = [len(self.s_collections[s_i].data) for s_i in s_inds]

        # Setup optimizer
        parameters = self.trainable_parameters(s_inds)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0], **adam_params)

        # Setup everything for logging
        epoch_elapsed_time = np.zeros(n_epochs)
        epoch_nll = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_p_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_u_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_nelbo = np.zeros(n_epochs)

        # Perform fitting
        prev_learning_rate = learning_rate_values[0]
        next_post_ri_ind = 0 # Index of next subject to re-initialize posteriors for
        for e_i in range(n_epochs):

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= e_i)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind]
            if cur_learning_rate != prev_learning_rate:
                # We reset the whole optimizer because ADAM is an adaptive optimizer
                optimizer = torch.optim.Adam(parameters, lr=cur_learning_rate, **adam_params)
                prev_learning_rate = cur_learning_rate

            # Setup the iterators to go through the current data for this epoch in a random order
            epoch_data_iterators = [dl.__iter__() for dl in data_loaders]

            # Re-initialize a posterior if we are suppose to
            if (e_i % post_reinit_int == 0) and (e_i != 0):
                ri_s_ind = s_inds[next_post_ri_ind]

                s_coll = self.s_collections[ri_s_ind]

                ri_p_dists = s_coll.p_dists
                ri_u_dists = s_coll.u_dists

                for g, d in enumerate(ri_p_dists):
                    if not isinstance(d, torch.Tensor):
                        s_coll.p_ri_fcns[g](d)

                for h, d in enumerate(ri_u_dists):
                    if not isinstance(d, torch.Tensor):
                        s_coll.u_ri_fcns[h](d)

                print('*****************************************************')
                print('Start of Epoch: ' + str(e_i) + ', Reinitialized posteriors for subject index: ' + str(ri_s_ind))

                # Determine which subject we will re-initialize posteriors for next
                next_post_ri_ind += 1
                if next_post_ri_ind == n_fit_subjects:
                    next_post_ri_ind = 0

            # Process each batch
            for b_i in range(n_batches):

                batch_obj_log = 0
                # Zero gradients
                optimizer.zero_grad()

                batch_nll = np.zeros(n_fit_subjects)
                batch_sub_p_kl = np.zeros(n_fit_subjects)
                batch_sub_u_kl = np.zeros(n_fit_subjects)
                for i, s_i in enumerate(s_inds):

                    s_coll = self.s_collections[s_i]

                    # Get the data for this batch for this subject
                    batch_data = epoch_data_iterators[i].next()

                    # Send the data to the GPU if needed
                    batch_data.to(device=s_coll.device, non_blocking=s_coll.device.type == 'cuda')
                    # Form x and y for the batch
                    batch_x = [batch_data.data[i_g][batch_data.i_x,:] for i_g in s_coll.input_grps]
                    batch_y = [batch_data.data[i_h][batch_data.i_y,:] for i_h in s_coll.output_grps]
                    n_batch_data_pts = batch_x[0].shape[0]

                    # Sample the posterior distributions of modes for this subject
                    q_p_modes = [d if isinstance(d, torch.Tensor)
                                 else d.sample(s_coll.props[s_coll.input_props[g]])
                                 for g, d in enumerate(s_coll.p_dists)]

                    q_u_modes = [d if isinstance(d, torch.Tensor)
                                 else d.sample(s_coll.props[s_coll.output_props[h]])
                                 for h, d in enumerate(s_coll.u_dists)]

                    q_p_modes_standard = [smp if isinstance(s_coll.p_dists[g], torch.Tensor)
                                          else s_coll.p_dists[g].form_standard_sample(smp)
                                          for g, smp in enumerate(q_p_modes)]

                    q_u_modes_standard = [smp if isinstance(s_coll.u_dists[h], torch.Tensor)
                                          else s_coll.u_dists[h].form_standard_sample(smp)
                                          for h, smp in enumerate(q_u_modes)]

                    # Make sure the m module is on the correct device for this subject, this is
                    # important when subject models share an m function
                    s_coll.s_mdl.m = s_coll.s_mdl.m.to(s_coll.device)

                    # Calculate the conditional log-likelihood for this subject
                    y_pred = s_coll.s_mdl.cond_forward(x=batch_x, p=q_p_modes_standard, u=q_u_modes_standard)
                    nll = (float(n_smp_data_points[i])/n_batch_data_pts)*s_coll.s_mdl.neg_ll(y=batch_y, mn=y_pred)
                    nll.backward(retain_graph=True)
                    batch_obj_log += nll.detach().cpu().numpy()

                    # Calculate KL diverengences between posteriors on modes and priors for this subject
                    s_p_kl = 0
                    for g, d in enumerate(self.p_priors):
                        if d is not None:
                            s_p_kl += torch.sum(s_coll.p_dists[g].kl(d_2=d, x=s_coll.props[s_coll.input_props[g]],
                                                           smp=q_p_modes[g]))
                            s_p_kl.backward()
                            batch_obj_log += s_p_kl.detach().cpu().numpy()

                    s_u_kl = 0
                    for h, d in enumerate(self.u_priors):
                        if d is not None:
                            s_u_kl += torch.sum(s_coll.u_dists[h].kl(d_2=d, x=s_coll.props[s_coll.output_props[h]],
                                                           smp=q_u_modes[h]))
                            s_u_kl.backward()
                            batch_obj_log += s_u_kl.detach().cpu().numpy()

                    # Record the log likelihood and kl divergences for each fit model for logging (we currently only
                    # save this for the last batch in an epoch
                    if b_i == n_batches - 1:
                        batch_nll[i] = nll.detach().cpu().numpy()
                        batch_sub_p_kl[i] = s_p_kl.detach().cpu().numpy()
                        batch_sub_u_kl[i] = s_u_kl.detach().cpu().numpy()

                # Take a gradient step
                optimizer.step()

                # Make sure no private variance values are too small
                with torch.no_grad():
                    for s_j in s_inds:
                        s_coll = self.s_collections[s_j]
                        s_min_var = s_coll.min_var
                        s_mdl = s_coll.s_mdl
                        for h in range(s_mdl.n_output_groups):
                            small_psi_inds = torch.nonzero(s_mdl.psi[h] < s_min_var[h])
                            s_mdl.psi[h].data[small_psi_inds] = s_min_var[h]

            # Take care of logging everything
            elapsed_time = time.time() - t_start
            epoch_elapsed_time[e_i] = elapsed_time
            epoch_nll[e_i,:] = batch_nll
            epoch_sub_p_kl[e_i,:] = batch_sub_p_kl
            epoch_sub_u_kl[e_i,:] = batch_sub_u_kl
            epoch_nelbo[e_i] = batch_obj_log

            if e_i % update_int == 0:
                print('*****************************************************')
                print('Epoch ' + str(e_i) + ' complete.  Obj: ' +
                      '{:.2e}'.format(float(batch_obj_log)) +
                      ', LR: '  + str(cur_learning_rate))
                if print_mdl_nlls:
                    print(format_output_list(base_str='Model NLLs: ', it_str='s_', vls=batch_nll, inds=s_inds))
                if print_sub_kls:
                    print(format_output_list(base_str='Subj P KLs: ', it_str='s_', vls=batch_sub_p_kl, inds=s_inds))
                    print(format_output_list(base_str='Subj U KLs: ', it_str='s_', vls=batch_sub_u_kl, inds=s_inds))
                if print_memory_usage:
                    device_memory_usage = self.get_device_memory_usage()
                    print(format_output_list(base_str='Device memory usage: ', it_str='d_',
                          vls=device_memory_usage, inds=range(len(device_memory_usage))))
                    device_max_memory_usage = self.get_device_max_memory_allocated()
                    print(format_output_list(base_str='Device max memory usage: ', it_str='d_',
                          vls=device_max_memory_usage, inds=range(len(device_max_memory_usage))))


                print('Elapsed time: ' + str(elapsed_time))

        # Return logs
        log = {'elapsed_time': epoch_elapsed_time, 'mdl_nll': epoch_nll, 'sub_p_kl': epoch_sub_p_kl,
               'sub_u_kl': epoch_sub_u_kl, 'nelbo': epoch_nelbo}
        return log



