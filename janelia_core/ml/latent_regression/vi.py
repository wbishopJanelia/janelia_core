""" Tools for fitting latent regression models with variational inference. """

import itertools
import time
from typing import List, Sequence, Union


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


class SubjectVICollection():
    """ Holds data, likelihood models and posteriors for fitting data to a single subject with variational inference."""

    def __init__(self, s_mdl: LatentRegModel, p_dists: Sequence, u_dists: Sequence,
                 data: Union[TimeSeriesBatch, TimeSeriesDataset], input_grps: Sequence, output_grps: Sequence,
                 props: Sequence, input_props: Sequence, output_props: Sequence, min_var: Sequence[float]):
        """ Creates a new SubjectVICollection object.

        Args:
            s_mdl: The likelihood model for the subject.

            u_dists: The posterior distributions for the u modes.  u_modes[h] is either:
                1) A janelia_core.ml.torch_distributions import CondVAEDistriubtion
                2) A torch tensor (if there is no distribution for the u modes for group h

            p_dists: The posterior distributions for the p modes.  Same form as u_dists.

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
        if self.p_priors is not None:
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

        if self.p_priors is not None:
            p_params = itertools.chain(*[d.parameters() for d in self.p_priors if d is not None])
            u_params = itertools.chain(*[d.parameters() for d in self.u_priors if d is not None])
        else:
            p_params = []
            u_params = []

        collection_params = itertools.chain(*[self.s_collections[s_i].trainable_parameters() for s_i in s_inds])

        all_params = list(itertools.chain(p_params, u_params, collection_params))

        # Clean up duplicate parameters.  Subject models might shared a posterior, for example, so we need to
        # check for this
        non_duplicate_params = set(all_params)
        return list(non_duplicate_params)

    def generate_data_loaders(self, n_batches: int, s_inds: Sequence[int] = None,
                              pin_memory: bool = False, num_workers: int = 2) -> Sequence:
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

            num_workers: The number of workers to use for data loaders

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
                                            shuffle=True, num_workers=num_workers)
                for i, s_i in enumerate(s_inds)]

    def get_device_memory_usage(self) -> Sequence:
        """ Returns the memory usage of each device in the fitter.

        The memory usage for cpu devices will be nan.

        Returns:
            m_usage: m_usage[i] is the amount of memory used on the i^th device
        """

        if self.p_priors is not None:
            all_devices = set(self.s_collection_devices + [self.prior_device])
        else:
            all_devices = self.s_collection_devices

        return [torch.cuda.memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in all_devices]

    def get_device_max_memory_allocated(self) -> Sequence:
        """ Returns the max memory usage of each device in the fitter.

        The memory usage for cpu devices will be nan.

        Returns:
            m_usage: m_usage[i] is the max amount of memory used on the i^th device
        """

        if self.p_priors is not None:
            all_devices = set(self.s_collection_devices + [self.prior_device])
        else:
            all_devices = self.s_collection_devices

        return [torch.cuda.max_memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in all_devices]

    def fit(self, n_epochs: int = 10, n_batches: int = 10, learning_rates = .01,
            adam_params: dict = {}, s_inds: Sequence[int] = None, pin_memory: bool = False,
            weight_penalty: float = 0.0, weight_penalty_type = 'l2',
            enforce_priors: bool = True, sample_posteriors: bool = True,
            num_data_loader_workers: int = 2,
            update_int: int = 1, print_mdl_nlls: bool = True, print_sub_kls: bool = True,
            print_memory_usage: bool = True, print_sub_weight_penalties = True):
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

            weight_penalty: If not 0, a penalty can be applied to the sampled weights for each subject.  See the
            description of weight_penalty_type below for more information.

            weight_penalty_type: We apply penalties to the sampled modes for each subject of the form
            \sum_g weight_penalty*||P_g|| + \sum_h weight_penalty*||U_h|| where ||o|| is a either the l1 or l2 norm.
            The value of weight_penalty_type determines the type of norm.

            enforce_priors: If enforce priors is true, the KL between priors and posteriors is included in the
            objective to be optimized (i.e., standard variational inference).  If False, the KL term is ignored
            and only the expected negative log-likelihood term in the ELBO is optimized.

            sample_posteriors: If true, posteriors will be sampled when fitting.  This is required for variational
            inference.  However, if false, the mean of the posteriors will be used as the sample.  In this case,
            this is equivalent to using a single function (the posterior mean) to set the loadings for each subject.
            This may be helpful for initialization.  Note that if sample_posteriors is false, enforce_priors must
            also be false.

            num_data_loader_workers: The number of works to use when creating data loaders.

            update_int: Fitting status will be printed to screen every update_int number of epochs

            print_mdl_nlls: If true, when fitting status is printed to screen, the negative log likelihood of each
            evaluated model will be printed to screen.

            print_sub_kls: If true, when fitting status is printed to screen, the kl divergence for the p and u modes
            for each fit subject will be printed to screen.

            print_sub_weight_penalties: If true, when fitting status is printed to screen, the weight penalties for
            each fit subject will be printed to screen.

            print_memory_usage: If true, when fitting status is printed to screen, the memory usage of each
            device will be printed to streen.

        Return:
            log: A dictionary with the following entries:

                'elapsed_time': A numpy array of the elapsed time for completion of each epoch

                'mdl_nll': mdl_nll[e, i] is the negative log likelihood for the subject model s_inds[i] at the start
                of epoch e (that is when the objective has been calculated but before parameters have been updated)

                'sub_p_kl': sub_p_kl[e,i] is the kl divergence between the posterior and conditional prior for the p
                modes for subject i at the start of epoch e.

                'sub_u_kl': sub_u_kl[e,i] is the kl divergence between the posterior and conditional prior the u modes
                for subject i at the start of epoch e.

                'sub_weight_penalties': sub_weight_penalties[e, i] is the weight penalty for subject i at the start
                of epoch e.

                obj: obj[e] contains the objective value at the start of epoch e.  This is the negative evidence lower
                bound + weight penalties.

        Raises:
            RuntimeError: If distribute() has not been called before fitting.
            ValueError: If sample_posteriors is False but enforce_priors is True.
            ValueError: If weight_penalty_type is not 'l1' or 'l2'.

        """

        if not self.distributed:
            raise(RuntimeError('self.distribute() must be called before fitting.'))
        if (not sample_posteriors) and enforce_priors:
            raise(ValueError('If sample posteriors is false, enforce_priors must also be false.'))

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
        data_loaders = self.generate_data_loaders(n_batches=n_batches, s_inds=s_inds, pin_memory=pin_memory,
                                                  num_workers=num_data_loader_workers)
        n_smp_data_points = [len(self.s_collections[s_i].data) for s_i in s_inds]

        # Setup optimizer
        parameters = self.trainable_parameters(s_inds)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate_values[0], **adam_params)

        # Setup everything for logging
        epoch_elapsed_time = np.zeros(n_epochs)
        epoch_nll = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_p_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_u_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_weight_penalties = np.zeros([n_epochs, n_fit_subjects])
        epoch_obj = np.zeros(n_epochs)

        # Perform fitting
        prev_learning_rate = learning_rate_values[0]
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

            # Process each batch
            for b_i in range(n_batches):

                batch_obj_log = 0
                # Zero gradients
                optimizer.zero_grad()

                batch_nll = np.zeros(n_fit_subjects)
                batch_sub_p_kl = np.zeros(n_fit_subjects)
                batch_sub_u_kl = np.zeros(n_fit_subjects)
                batch_sub_weight_penalties = np.zeros(n_fit_subjects)
                for i, s_i in enumerate(s_inds):

                    s_coll = self.s_collections[s_i]

                    # Get the data for this batch for this subject
                    batch_data = epoch_data_iterators[i].next()

                    # Send the data to the GPU if needed
                    batch_data.to(device=s_coll.device, non_blocking=s_coll.device.type == 'cuda')

                    # Form x and y for the batch
                    batch_x = [batch_data.data[i_g][batch_data.i_x, :] for i_g in s_coll.input_grps]
                    batch_y = [batch_data.data[i_h][batch_data.i_y, :] for i_h in s_coll.output_grps]
                    n_batch_data_pts = batch_x[0].shape[0]

                    # Make sure the posterior is on the right GPU for this subject (important if we are
                    # using a shared posterior)
                    for d in s_coll.p_dists:
                        if not isinstance(d, torch.Tensor):
                            d.to(s_coll.device)
                    for d in s_coll.u_dists:
                        if not isinstance(d, torch.Tensor):
                            d.to(s_coll.device)

                    # Sample the posterior distributions of modes for this subject
                    if sample_posteriors:
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
                    else:
                        q_p_modes_standard = [d if isinstance(d, torch.Tensor)
                                              else d(s_coll.props[s_coll.input_props[g]])
                                              for g, d in enumerate(s_coll.p_dists)]

                        q_u_modes_standard = [d if isinstance(d, torch.Tensor)
                                              else d(s_coll.props[s_coll.output_props[h]])
                                              for h, d in enumerate(s_coll.u_dists)]

                    # Make sure the m module is on the correct device for this subject, this is
                    # important when subject models share an m function
                    s_coll.s_mdl.m = s_coll.s_mdl.m.to(s_coll.device)

                    # Calculate the conditional log-likelihood for this subject
                    y_pred = s_coll.s_mdl.cond_forward(x=batch_x, p=q_p_modes_standard, u=q_u_modes_standard)
                    nll = (float(n_smp_data_points[i])/n_batch_data_pts)*s_coll.s_mdl.neg_ll(y=batch_y, mn=y_pred)
                    nll.backward(retain_graph=True)
                    batch_obj_log += nll.detach().cpu().numpy()

                    # Apply penalties to modes for this subject if we are suppose to
                    if weight_penalty != 0:
                        s_w_pen = torch.zeros([1], device=s_coll.device)[0]  # Weird indexing is to get a scalar tensor

                        # Calculate penalties for p modes
                        for g, mode_g in enumerate(q_p_modes_standard):
                            if not isinstance(s_coll.p_dists[g], torch.Tensor):
                                if weight_penalty_type == 'l1':
                                    s_w_pen += weight_penalty*torch.sum(torch.abs(mode_g))
                                elif weight_penalty_type == 'l2':
                                    s_w_pen += weight_penalty*torch.sum(mode_g**2)
                                else:
                                    raise(ValueError('weight_penalty type must be either l1 or l2'))

                        # Calculate penalties for u modes
                        for h, mode_h in enumerate(q_u_modes_standard):
                            if not isinstance(s_coll.u_dists[h], torch.Tensor):
                                if weight_penalty_type == 'l1':
                                    s_w_pen += weight_penalty*torch.sum(torch.abs(mode_h))
                                elif weight_penalty_type == 'l2':
                                    s_w_pen += weight_penalty*torch.sum(mode_h**2)
                                else:
                                    raise(ValueError('weight_penalty type must be either l1 or l2'))

                        s_w_pen.backward()
                        batch_obj_log += s_w_pen.detach().cpu().numpy()

                    # Calculate KL diverengences between posteriors on modes and priors for this subject
                    if enforce_priors:
                        s_p_kl = torch.zeros([1], device=s_coll.device)[0]  # Weird indexing is to get a scalar tensor
                        for g, d in enumerate(self.p_priors):
                            if d is not None:
                                s_p_kl += torch.sum(s_coll.p_dists[g].kl(d_2=d, x=s_coll.props[s_coll.input_props[g]],
                                                                         smp=q_p_modes[g]))
                                s_p_kl.backward()
                                batch_obj_log += s_p_kl.detach().cpu().numpy()

                        s_u_kl = torch.zeros([1], device=s_coll.device)[0]  # Weird indexing is to get a scalar tensor
                        for h, d in enumerate(self.u_priors):
                            if d is not None:
                                s_u_kl += torch.sum(s_coll.u_dists[h].kl(d_2=d, x=s_coll.props[s_coll.output_props[h]],
                                                                         smp=q_u_modes[h]))
                                s_u_kl.backward()
                                batch_obj_log += s_u_kl.detach().cpu().numpy()

                    # Record the log likelihood, kl divergences and weight penalties for each subject for logging
                    batch_nll[i] = nll.detach().cpu().numpy()
                    if weight_penalty != 0:
                        batch_sub_weight_penalties[i] = s_w_pen.detach().cpu().numpy()
                    if enforce_priors:
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
            epoch_nll[e_i, :] = batch_nll
            epoch_sub_p_kl[e_i, :] = batch_sub_p_kl
            epoch_sub_u_kl[e_i, :] = batch_sub_u_kl
            epoch_sub_weight_penalties[e_i, :] = batch_sub_weight_penalties
            epoch_obj[e_i] = batch_obj_log

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
                if print_sub_weight_penalties:
                    print(format_output_list(base_str='Subj ' + weight_penalty_type + ' W Penalties: ', it_str='s_',
                                             vls=batch_sub_weight_penalties, inds=s_inds))
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
               'sub_u_kl': epoch_sub_u_kl, 'sub_weight_penalties': epoch_sub_weight_penalties, 'obj': epoch_obj}
        return log


def predict(s_collection: SubjectVICollection, data: TimeSeriesBatch, batch_size: int = 100) -> List[torch.tensor]:
    """ Predicts output given input from a model with posterior distributions over modes.

    When predicting output, the posterior mean for each mode is used.

    Note: All predictions will be returned on host (cpu) memory as numpy arrays.

    Args:
        s_collection: The collection for the subject.   Any data in the collection will be ignored.

        data: The data to predict with.

        batch_size: The number of samples we predict on at a time.  This is helpful if using a GPU
        with limited memory.

    Returns:
        pred_mn: The predicted means given the input.
    """

    n_total_smps = len(data)
    n_batches = int(np.ceil(float(n_total_smps)/batch_size))

    # Get the posterior means for the modes
    q_p_modes = [d if isinstance(d, torch.Tensor)
                 else d(s_collection.props[s_collection.input_props[g]])
                 for g, d in enumerate(s_collection.p_dists)]

    q_u_modes = [d if isinstance(d, torch.Tensor)
                 else d(s_collection.props[s_collection.output_props[h]])
                 for h, d in enumerate(s_collection.u_dists)]

    y = [None]*n_batches
    batch_start = 0
    for b_i in range(n_batches):
        batch_end = batch_start + batch_size
        batch_data = data[batch_start:batch_end]

        # Move data to device the collection is on
        batch_data.to(s_collection.device, non_blocking=s_collection.device.type == 'cuda')

        # Form x
        batch_x = [batch_data.data[i_g][batch_data.i_x] for i_g in s_collection.input_grps]
        with torch.no_grad():
            batch_y = s_collection.s_mdl.cond_forward(x=batch_x, p=q_p_modes, u=q_u_modes)
        batch_y = [t.cpu().numpy() for t in batch_y]
        y[b_i] = batch_y

        batch_start = batch_end

    # Concatenate output
    n_output_grps = len(y[0])
    y_out = [None]*n_output_grps
    for h in range(n_output_grps):
        y_out[h] = np.concatenate([batch_y[h] for batch_y in y])

    return y_out
