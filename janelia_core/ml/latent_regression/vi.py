""" Tools for fitting latent regression models with variational inference. """

import copy
import itertools
import time
from typing import Callable, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

import janelia_core
from janelia_core.ml.datasets import TimeSeriesBatch
from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.torch_distributions import CondMatrixProductDistribution
from janelia_core.ml.torch_distributions import CondVAEDistribution
from janelia_core.ml.torch_distributions import DistributionPenalizer
from janelia_core.ml.torch_parameter_penalizers import ParameterPenalizer

from janelia_core.ml.utils import format_and_check_learning_rates
from janelia_core.ml.utils import torch_devices_memory_usage


def format_output_list(base_str: str, it_str: str, vls: Sequence[float], inds: Sequence[int]):
    """ Produces a string to display a list of outputs.

    String will be of the format:

        base_str + ' ' + it_str+inds[0] + ': ' + str(vls[0]) + ', ' + it_str+inds[1] + ': ' + str(vls[1]) + ...

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


def concatenate_check_points(check_points: Sequence[dict], params: Sequence[dict]) -> List:
    """ Concatenates lists of checkpoints and computes total epochs to each checkpoint from multiple rounds of fitting.

    Args:
        check_points: check_points[i] is a sequence of check points produced by a call to
        MultiSubjectVIFitter.fit().  It is assumed that entries in check_points match the order they were produced
        in the actual fitting.

        params: params[i] is a dictionary for the call to MultiSubjectVIFitter.fit() that produced the checkpoints
        in check_points[i].  Is should have the fields:
            cp_epochs: For the epochs that the checkpoints were created at
            n_epochs: For the total number of epochs that fitting was run for

    Returns:

        conc_check_points: The concatenated list of check points

        cp_epochs: The total accumulated epochs that were run to get to each check point.
    """

    n_fits = len(check_points)

    n_fit_epochs = [copy.deepcopy(d['n_epochs']) for d in params]
    cp_epochs = [copy.deepcopy(d['cp_epochs']) for d in params]

    # Make sure cp_epochs are arrays
    cp_epochs = [np.asarray(vls) for vls in cp_epochs]

    total_sum = 0
    for f_i in range(n_fits):
        cp_epochs[f_i] += total_sum
        total_sum += n_fit_epochs[f_i]

    cp_epochs = np.concatenate(cp_epochs)

    conc_check_points = list(itertools.chain(*check_points))

    return [conc_check_points, cp_epochs]


class PriorCollection():
    """ Holds prior distributions when fitting models with variational inference.

    This object offers convenience functions for getting all the parameters for the priors as well as moving the
    distributions to different devices.
    """
    def __init__(self, p_dists: Sequence[Union[CondVAEDistribution, None]],
                 u_dists: Sequence[Union[CondVAEDistribution, None]],
                 psi_dists: Sequence[Union[CondVAEDistribution, None]],
                 scale_dists: Union[Sequence[Union[CondVAEDistribution, None]], None] = None,
                 offset_dists: Union[Sequence[Union[CondVAEDistribution, None]], None] = None,
                 direct_mapping_dists: Union[Sequence[Union[CondVAEDistribution, None]], None] = None):
        """ Creates a new PriorCollection object.

        When specifying a prior distribution over a parameter, the user can specify either a CondVAEDistribution object,
        which will be used if posterior distributions are fit over the parameter.  Alternatively, if point estimates
        will be fit over the parameter, the user can provide the value None.  E.g., if a distribution will be fit over
        the first input group modes but not the second p_dists would be set to [d, None], where d is a
        CondVAEDistribution object.

        Args:
            p_dists: Prior distributions over p modes.

            u_dists: Prior distributions over u modes.

            psi_dists: Prior distributions over variance parameters.

            scale_dists: Prior distributions over scale parameters.  If scale parameters are not used in subject
            models, set this to None.

            offset_dists: Prior distributions over offset parameters.  If offset parameters are not use in subject
            models, set this to None.

            direct_mapping_dists: Prior distributions over direct mappings.  direct_mapping_dists[i] should be the
            prior over the direct mappings in direct_pairs[i] in subject  models.  If direct mappings are not used,
            set this to None.

        """

        self.p_dists = p_dists
        self.u_dists = u_dists
        self.psi_dists = psi_dists
        self.scale_dists = scale_dists
        self.offset_dists = offset_dists
        self.direct_mapping_dists = direct_mapping_dists

    def get_used_devices(self):
        """ Returns all devices that priors are on. """
        p_dist_devices = [next(d.parameters())[0].device for d in self.p_dists if d is not None]
        u_dist_devices = [next(d.parameters())[0].device for d in self.u_dists if d is not None]
        psi_dist_devices = [next(d.parameters())[0].device for d in self.psi_dists if d is not None]

        if self.scale_dists is not None:
            scale_dist_devices = [next(d.parameters())[0].device for d in self.scale_dists if d is not None]
        else:
            scale_dist_devices = []

        if self.offset_dists is not None:
            offset_dist_devices = [next(d.parameters())[0].device for d in self.offset_dists if d is not None]
        else:
            offset_dist_devices = []

        if self.direct_mapping_dists is not None:
            direct_mapping_dist_devices = [next(d.parameters())[0].device
                                           for d in self.direct_mapping_dists if d is not None]
        else:
            direct_mapping_dist_devices = []

        return list(set(p_dist_devices + u_dist_devices + psi_dist_devices + scale_dist_devices +
                        offset_dist_devices + direct_mapping_dist_devices))

    def r_params(self) -> List[torch.nn.parameter.Parameter]:
        """ Gets parameters of all modules for which gradients can be estimated with the reparameterization trick. """

        p_dist_params = itertools.chain(*[d.r_params() for d in self.p_dists if d is not None])
        u_dist_params = itertools.chain(*[d.r_params() for d in self.u_dists if d is not None])
        psi_dist_params = itertools.chain(*[d.r_params() for d in self.psi_dists if d is not None])

        if self.scale_dists is not None:
            scale_dist_params = itertools.chain(*[d.r_params() for d in self.scale_dists if d is not None])
        else:
            scale_dist_params = []

        if self.offset_dists is not None:
            offset_dist_params = itertools.chain(*[d.r_params() for d in self.offset_dists if d is not None])
        else:
            offset_dist_params = []

        if self.direct_mapping_dists is not None:
            direct_mapping_dist_params = itertools.chain(*[d.r_params() for
                                                           d in self.direct_mapping_dists if d is not None])
        else:
            direct_mapping_dist_params = []

        return list(itertools.chain(p_dist_params, u_dist_params, psi_dist_params, scale_dist_params,
                                    offset_dist_params, direct_mapping_dist_params))

    def s_params(self) -> List[torch.nn.parameter.Parameter]:
        """ Gets parameters of all modules for which gradients can be estimated with the score method. """

        p_dist_params = itertools.chain(*[d.s_params() for d in self.p_dists if d is not None])
        u_dist_params = itertools.chain(*[d.s_params() for d in self.u_dists if d is not None])
        psi_dist_params = itertools.chain(*[d.s_params() for d in self.psi_dists if d is not None])

        if self.scale_dists is not None:
            scale_dist_params = itertools.chain(*[d.s_params() for d in self.scale_dists if d is not None])
        else:
            scale_dist_params = []

        if self.offset_dists is not None:
            offset_dist_params = itertools.chain(*[d.s_params() for d in self.offset_dists if d is not None])
        else:
            offset_dist_params = []

        if self.direct_mapping_dists is not None:
            direct_mapping_dist_params = itertools.chain(*[d.s_params() for
                                                           d in self.direct_mapping_dists if d is not None])
        else:
            direct_mapping_dist_params = []

        return list(itertools.chain(p_dist_params, u_dist_params, psi_dist_params, scale_dist_params,
                                    offset_dist_params, direct_mapping_dist_params))

    def to(self, device: Union[torch.device, int]):
        """ Moves all distributions in the collection to a specified device. """

        def move_if_not_none(dists):
            for d in dists:
                if d is not None:
                    d.to(device)

        move_if_not_none(self.p_dists)
        move_if_not_none(self.u_dists)
        move_if_not_none(self.psi_dists)

        if self.scale_dists is not None:
            move_if_not_none(self.scale_dists)

        if self.offset_dists is not None:
            move_if_not_none(self.offset_dists)

        if self.direct_mapping_dists is not None:
            move_if_not_none(self.direct_mapping_dists)

# TODO: Remove if we confirm we no longer need
#def compute_prior_penalty(mn: torch.Tensor, positions: torch.Tensor):
#    """ Computes a penalty for a sampled prior.  This is experimental.
#
#    Args:
#    """
#
#    n_modes = mn.shape[1]
#    compute_device = mn.device
#
#    penalty = torch.zeros([1], device=compute_device)[0]  # Weird indexing is to get a scalar tensor
#    for m_i in range(n_modes):
#        mode_abs_vls = torch.sum(torch.abs(mn[:, m_i:m_i+1]))
#        mode_l2_norm_sq = torch.sum(mn[:, m_i:m_i+1]**2)
#        #mode_weighted_positions = positions*mode_abs_vls
#        #mode_weighted_center = torch.mean(mode_weighted_positions, dim=0)
#        #penalty += torch.sum(torch.sum((positions - mode_weighted_center)**2, dim=1)*torch.squeeze(mode_abs_vls))
#        penalty += mode_abs_vls + (mode_l2_norm_sq - 10)**2
#    return penalty


class SubjectVICollection():
    """ Holds data, likelihood models and posteriors for fitting data to a single subject with variational inference.

    This object offers convenience functions to get all the trainable parameters for a subject model and its posteriors
    as well as moving everything needed for fitting for that subject to different devices.
    """

    def __init__(self, s_mdl: LatentRegModel, p_dists: Sequence[Union[CondVAEDistribution, None]],
                 u_dists: Sequence[Union[CondVAEDistribution, None]],
                 psi_dists: Sequence[Union[CondVAEDistribution, None]],
                 data: TimeSeriesBatch, input_grps: Sequence[int], output_grps: Sequence[int],
                 props: Union[Sequence[torch.Tensor], None], p_props: Sequence[int], u_props: Sequence[int],
                 psi_props: Sequence[int],
                 scale_dists: Union[
                     Sequence[Union[CondVAEDistribution, None]], None] = None,
                 offset_dists: Union[
                     Sequence[Union[CondVAEDistribution, None]], None] = None,
                 direct_mappings_dists: Union[
                     Sequence[Union[CondVAEDistribution, None]], None] = None,
                 scale_props: Union[Sequence[int], None] = None, offset_props: Union[Sequence[int], None] = None,
                 direct_mapping_props: Union[Sequence[int], None] = None, min_var: Union[Sequence[float], None] = None):
        """ Creates a new SubjectVICollection object.

        When specify the distribution over a parameter of the model, the user can specify either (1) a
        janelia_core.ml.torch_distributions.CondVAEDistribution object or (2) None.  Specifying a distribution
        means that distribution will be used as the posterior distribution over the parameter for fitting.  Specifying
        None, means that a distribution won't be fit for that parameter.  Instead, a point estimate will be fit by
        directly optimizing the value of the parameter in the subject model.

        Args:
            s_mdl: The likelihood model for the subject.

            p_dists: The posterior distributions for the p modes.

            u_dists: The posterior distributions for the u modes.

            psi_dists: The posterior distributions for psi parameters.

            data: Data for the subject.

            input_grps: input_grps[g] is the index into data.data for the g^th input group

            output_grps: output_grps[h] is the index into data.data for the h^th output group

            props: props[i] is a tensor of properties.  If there are no properties, this should be None.

            p_props: p_props[g] is the index into props for the properties for the modes for the g^th input group.  If
            there are no distributions over the modes for this group, p_props[g] should be None. p_props[g] should also
            be None if there are distributions for these modes but they are not conditioned on anything.

            u_props:  p_props[h] is the index into props for the properties for the modes for the h^th output group,
            same format as u_props.

            psi_props:  psi_props[h] is the index into props for the properties for the variances for the h^th output
            group, same format as u_props.

            scale_dists: The posterior distributions for scale parameters. If scales are not used in the model,
            set to None.

            offset_dists: The posterior distributions for offset parameters, same form as scale_dists.

            direct_mappings_dists: Distributions over direct mappings.  If there are no direct mappings in the model,
            this should be None.  If there are direct mappings, direct_mapping_dists[i] is the distribution over
            s_mdl.direct_mappings[i].

            scale_props:  scale_props[h] is the index into props for the properties for the scales for the h^th output
            group, same format as u_props.

            offset_props:  offset_props[h] is the index into props for the properties for the offsets for the h^th output
            group, same format as u_props.

            direct_mapping_props: direct_mapping_props[i] is the index into props for the properties for the i^th
            direct mapping, same format as u_props

            min_var: min_var[h] is the minimum variance for the additive noise variables for output group h. If
            set to None, min_var for all output groups will be set to .01.

        """

        self.s_mdl = s_mdl
        self.p_dists = p_dists
        self.u_dists = u_dists
        self.scale_dists = scale_dists
        self.offset_dists = offset_dists
        self.psi_dists = psi_dists
        self.direct_mapping_dists = direct_mappings_dists
        self.data = data
        self.input_grps = input_grps
        self.output_grps = output_grps
        self.props = props
        self.u_props = u_props
        self.p_props = p_props
        self.scale_props = scale_props
        self.offset_props = offset_props
        self.psi_props = psi_props
        self.direct_mapping_props = direct_mapping_props

        if min_var is None:
            min_var = [.01]*len(p_dists)
        self.min_var = min_var

    def trainable_parameters(self) -> List[torch.nn.parameter.Parameter]:
        """ Returns all trainable parameters for the collection.

        Returns:
            params: The list of parameters.
        """

        s_mdl_parameters = self.s_mdl.trainable_parameters()

        p_dist_params = itertools.chain(*[d.parameters() for d in self.p_dists if d is not None])
        u_dist_params = itertools.chain(*[d.parameters() for d in self.u_dists if d is not None])
        psi_dist_params = itertools.chain(*[d.parameters() for d in self.psi_dists if d is not None])

        if self.scale_dists is not None:
            scale_dist_params = itertools.chain(*[d.parameters() for d in self.scale_dists if d is not None])
        else:
            scale_dist_params = []

        if self.offset_dists is not None:
            offset_dist_params = itertools.chain(*[d.parameters() for d in self.offset_dists if d is not None])
        else:
            offset_dist_params = []

        if self.direct_mapping_dists is not None:
            direct_mapping_dist_params = itertools.chain(*[d.parameters() for d in self.direct_mapping_dists
                                                           if d is not None])
        else:
            direct_mapping_dist_params = []

        return list(itertools.chain(s_mdl_parameters, p_dist_params, u_dist_params, psi_dist_params,
                                    scale_dist_params, offset_dist_params, direct_mapping_dist_params))

    def to(self, device: Union[torch.device, int], distribute_data: bool = False):
        """ Moves all relevant attributes of the collection to the specified device.

        Note that by default fitting data will not be moved to the device.

        Args:
            device: The device to move attributes to.

            distribute_data: True if fitting data should be moved to the device as well
        """

        self.s_mdl.to(device)

        def move_if_not_none(dists):
            for d in dists:
                if d is not None:
                    d.to(device)

        move_if_not_none(self.p_dists)
        move_if_not_none(self.u_dists)
        move_if_not_none(self.psi_dists)

        if self.scale_dists is not None:
            move_if_not_none(self.scale_dists)
        if self.offset_dists is not None:
            move_if_not_none(self.offset_dists)
        if self.direct_mapping_dists is not None:
            move_if_not_none(self.direct_mapping_dists)

        if self.props is not None:
            self.props = [p.to(device) for p in self.props]

        if distribute_data and self.data is not None:
            self.data.to(device)


class MultiSubjectVIFitter():
    """ Object for fitting a collection of latent regression models with variational inference.

    """

    def __init__(self, s_collections: Sequence[SubjectVICollection], prior_collection: PriorCollection,
                 input_modules: Union[torch.nn.ModuleList, None] = None,
                 penalizers: Union[Sequence[ParameterPenalizer], None] = None):
        """ Creates a new MultiSubjectVIFitter object.

        Args:
            s_collections: A set of SubjectVICollections to use when fitting data.

            prior_collection: The prior collection to use when fitting data.

            input_modules: Contains modules to apply to inputs before they are passed to subject models.  If None,
            no input modules will be used.  If a list, the g^th module is the module to apply to the input for
            input group g.  If no module should be applied to a group, the entry for that group should be None.

            penalizers: Parameter penalizers that will be used when fitting.
        """

        # If we are not using input_modules, we still create an input_modules list with all None entries, so the
        # rest of our code can be a bit simpler
        if input_modules is None:
            input_modules = torch.nn.ModuleList([None]*len(s_collections[0].input_grps))

        self.s_collections = s_collections
        self.input_modules = input_modules
        self.prior_collection = prior_collection
        self.penalizers = penalizers

        self.distributed = False  # Keep track of if we have distributed everything yet

    def create_check_point(self, inc_penalizers: bool = False) -> dict:
        """ Returns copies of subject vi collections, priors as well as (optionally) penalizer parameters.

        Args:
            inc_penalizers: True if copies of penalizer parameters should be returned.

        Returns:
            cp_dict: A dictionary with the following keys:

                s_collections: Copies of the subject collections, with data and properties removed

                prior_collection: Copy of the prior collection

                input_modules: Copy of input modules.

                parameter_penalizer_dicts: If penalizer parameters are requested, then parameter_penalizer_dicts[i] is
                the dictionary witch check point parameters for the i^th parameter penalizer, where the ordering of
                penalizers is the same as when parameter penalizers were provided at the time of the creation of the
                Fitter object.
        """

        orig_devices = self.get_used_devices()
        self.to('cpu', distribute_data=True)

        s_collections_copy = copy.deepcopy(self.s_collections)
        for s_coll in s_collections_copy:
            s_coll.data = None
            s_coll.props = None

        prior_collection_copy = copy.deepcopy(self.prior_collection)

        input_modules_copy = copy.deepcopy(self.input_modules)

        if inc_penalizers:
            if self.penalizers is not None:
                parameter_penalizer_dicts = [p.check_point() for p in self.penalizers]
            else:
                parameter_penalizer_dicts = None

        # Move subject collections back to devices
        self.distribute(devices=orig_devices, distribute_data=True)

        return {'s_collections': s_collections_copy,
                'prior_collection': prior_collection_copy,
                'input_modules': input_modules_copy,
                'parameter_penalizer_dicts': parameter_penalizer_dicts}

    def distribute(self, devices: Sequence[Union[torch.device, int]], s_inds: Sequence[int] = None,
                   distribute_data: bool = False):
        """ Distributes priors, subject collections and penalizers across devices.

        Args:
            devices: Devices that priors and subject collections should be distributed across.

            s_inds: Indices into self.s_collections for subject models which should be distributed across devices.
            If none, all subject models will be distributed.

            distribute_data: True if all training data should be distributed to devices.  If there is enough
            device memory, this can speed up fitting.  If not, set this to false, and batches of data will
            be sent to the device for each training iteration.

        """
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        n_devices = len(devices)
        n_dist_mdls = len(s_inds)

        # Distribute priors; by convention priors go onto first device
        self.prior_collection.to(devices[0])

        # Distribute input modules - these get moved from device to device as they need to process input
        # for each subject model, so where they go really doesn't matter, but by default they start on the first
        # device
        self.input_modules.to(devices[0])

        # We also put penalizers on first device
        if self.penalizers is not None:
            for penalizer in self.penalizers:
                penalizer.to(devices[0])

        # Distribute subject collections
        for i in range(n_dist_mdls):
            device_ind = i % n_devices
            self.s_collections[s_inds[i]].to(devices[device_ind], distribute_data=distribute_data)

        self.distributed = True

    def trainable_parameters(self, s_inds: Sequence[int] = None, get_prior_params: bool = True) -> [list, list]:
        """ Gets all trainable parameters for fitting.

        This function returns separate parameters for penalizer parameters and all other parameters, to faciltate
        applying different learning rates to the penalizer parameters.

        Args:
            s_inds: Specifies the indices of subjects that will be fit.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

            get_prior_params: True if parameters of priors should be included in the set of returned parameters

        Returns:
             base_params: Parameters of priors, posteriors, input modules and subject models

             penalizer_params: Parameters of any penalizers
        """

        # Get base parameters
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        if get_prior_params:
            prior_params = self.prior_collection.r_params()
        else:
            prior_params = []

        collection_params = itertools.chain(*[self.s_collections[s_i].trainable_parameters() for s_i in s_inds])
        base_params = list(itertools.chain(collection_params, prior_params))

        # Clean for duplicate parameters.  Subject models might shared a posterior, for example, so we need to
        # check for this
        non_duplicate_base_params = list(set(base_params))

        # Add in input module parameters to base parameters
        input_module_params = list(self.input_modules.parameters())
        non_duplicate_base_params = non_duplicate_base_params + input_module_params

        # Get penalizer parameters - the get_penalizer_params() already checks for duplicates
        non_duplicate_penalizer_params = self.get_penalizer_params()

        return [non_duplicate_base_params, non_duplicate_penalizer_params]

    def get_penalizer_params(self, keys: Union[str, Sequence[str]] = None) -> list:
        """ Returns a list of penalizer parameters, optionally filtering by key.

        Args:
            keys: If provided, either a string of a single key that returned parameters should match or a
            sequence of keys parameters can match.  Any parameters not matching the requested key(s), will
            not be returned.  If keys is None, all penalizer parameters will be returned.

        Returns:
            params: A list of the requested parameters

        """

        # First, if we have no penalizers, we just return an empyt list
        if self.penalizers is None:
            return []

        # Use all keys if user has not provided any
        if keys is None:
                keys = itertools.chain(*[p.list_param_keys() for p in self.penalizers if p is not None])
        elif isinstance(keys, str):
            keys = [keys]

        # Get requested parameters
        keys = list(set(keys))

        params = itertools.chain(*[itertools.chain(*[p.get_marked_params(key) for p in self.penalizers])
                                   for key in keys])
        params = list(set(list(params)))

        return params

    def generate_batch_smp_inds(self, n_batches: int, s_inds: Sequence[int] = None) -> List[List[np.ndarray]]:
        """ Generates indices of random mini-batches of samples for each subject.

        Args:
            n_batches: The number of batches to break the data up for each subject into.

            s_inds: Specifies the indices of subjects that will be fit.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, s_inds = range(n_subjects).

        Returns:
            batch_smp_inds: batch_smp_inds[i][j] is the sample indices for the j^th batch for subject s_inds[i]

        """
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        n_subjects = len(s_inds)

        n_smps = [len(self.s_collections[s_i].data) for s_i in s_inds]

        for i, n_s in enumerate(n_smps):
            if n_s < n_batches:
                raise(ValueError('Subject ' + str(s_inds[i]) + ' has only ' + str(n_s) + ' samples, while '
                      + str(n_batches) + ' batches requested.'))

        batch_sizes = [int(np.floor(float(n_s)/n_batches)) for n_s in n_smps]

        batch_smp_inds = [None]*n_subjects
        for i in range(n_subjects):
            subject_batch_smp_inds = [None]*n_batches
            perm_inds = np.random.permutation(n_smps[i])
            start_smp_ind = 0
            for b_i in range(n_batches):
                end_smp_ind = start_smp_ind+batch_sizes[i]
                subject_batch_smp_inds[b_i] = perm_inds[start_smp_ind:end_smp_ind]
                start_smp_ind = end_smp_ind
            batch_smp_inds[i] = subject_batch_smp_inds

        return batch_smp_inds

    def get_used_devices(self):
        """ Lists any device currently used for fitting.

        Returns:
            devices: The list of devices parameters are on in.
        """

        s_mdl_devices = [s_coll.s_mdl.trainable_parameters()[0].device for s_coll in self.s_collections]
        data_devices = [s_coll.data.data[0].device for s_coll in self.s_collections]  # TODO: Should check all data tensors
        prior_devices = self.prior_collection.get_used_devices()
        if self.penalizers is not None:
            penalizer_parameters = [list(penalizer.parameters()) for penalizer in self.penalizers]
            penalizer_devices = [parameters[0].device for parameters in penalizer_parameters if len(parameters) > 0]
        else:
            penalizer_devices = []

        input_module_devices = [next(m.parameters()).device for m in self.input_modules if m is not None]

        return list(set(s_mdl_devices + data_devices + prior_devices + input_module_devices + penalizer_devices))

    def fit(self, n_epochs: int = 10, n_batches: int = 10, learning_rates=.01,
            adam_params: dict = {}, s_inds: Sequence[int] = None, fix_priors: bool = False,
            enforce_priors: bool = True, update_int: int = 1, print_opts: dict = None,
            cp_epochs: Sequence[int] = None, cp_penalizers: bool = True) -> [dict, Union[List, None]]:
        """

        Args:

            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the training data up into per epoch.  When multiple subjects have
            different numbers of total training samples, the batch size for each subject will be selected so we go
            through the entire training set for each subject after processing n_batches each epoch.

            learning_rates: If a single number, this is the learning rate to use for all epochs and parameters.
            Alternatively, this can be a list of tuples.  Each tuple is of the form
            (epoch, base_lr, penalizer_lr_opts), where epoch is the epoch the learning rates come into effect on,
            base_lr is the learning rate for all parameters other than the penalizer parameters and penalizer_lr_opts
            is a dictionary with keys specifying penalizer parameter keys and values giving the learning rate
            for those parameters. Multiple tuples can be provided to give a schedule of learning rates.  Here is an
            example learning_rates: [(0, .001, {'fast': 1, 'slow', .1}), (100, .0001, {'fast': .1, 'slow', .01}] that
            starts with a base learning rates of .001, a learning rate of 1 for parameters of the penalizers that
            should be assigned a fast learning rate and a learning rate of .1 for parameter of the penalizers that
            should be assigned a slow learning rate.  At epoch 100, the learning rates are divided by 10 in this
            example.

            adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object.
            Note that if learning rate is specified here *it will be ignored.* (Use the learning_rates option instead).
            The options specified here will be applied to all parameters at all iterations.

            s_inds: Specifies the indices of subjects to fit to.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

            fix_priors: True if priors should be fixed and not changed during fitting

            enforce_priors: True if we should calculate kl divergences between priors and posteriors and include
            this in the objective.

            update_int: Fitting status will be printed to screen every update_int number of epochs

            print_opts: Options controlling what is printed to screen.  See _print_fitting_status()

            cp_epochs: A sequence of epochs after which a check point of the models (as well as optionally the
            penalizers will be made).  If no check points should be made, set this to None.

            cp_penalizers: True if penalizers should be included in the check points.

        Return:
            log: A dictionary with the following entries:

                'elapsed_time': A numpy array of the elapsed time for completion of each epoch

                'mdl_nll': mdl_nll[e, i] is the negative log likelihood for the subject model s_inds[i] at the start
                of epoch e (that is when the objective has been calculated but before parameters have been updated)

                'sub_p_kl': sub_p_kl[e,i] is the kl divergence between the posterior and conditional prior for the p
                modes for subject i at the start of epoch e.

                'sub_u_kl': sub_u_kl[e,i] is the kl divergence between the posterior and conditional prior the u modes
                for subject i at the start of epoch e.

                'p_prior_penalties': p_prior_penalties[e, :] is the penalty calculated for each group of p priors at the
                start of epoch e.

                'u_prior_penalties': u_prior_penalties[e, :] is the penalty calculated for each group of u priors at the
                start of epoch e.

                parameter_penalties: parameter_penalties[e, :] is the penalty calculated for each parameter penalizer,
                with the order of entries in each row corresponding to the order penalizers were provided when creating
                the Fitter object.

                obj: obj[e] contains the objective value at the start of epoch e.  This is the negative evidence lower
                bound + weight penalties.

            check_points: check_points[i] are model parameters for the i^th requested checkpoint. If no check points
            were requested this will be None.

        Raises:
            RuntimeError: If distribute() has not been called before fitting.
            ValueError: If sample_posteriors is False but enforce_priors is True.
            ValueError: If weight_penalty_type is not 'l1' or 'l2'.

        """

        if print_opts is None:
            print_opts = {'mdl_nll': True,
                          'sub_kls': True,
                          'memory_usage': True,
                          'penalties': True,
                          'penalizer_states': True}

        if not self.distributed:
            raise(RuntimeError('self.distribute() must be called before fitting.'))

        # See what devices we are using for fitting (this is so we can later query their memory usage)
        all_devices = self.get_used_devices()

        t_start = time.time()  # Get starting time

        # Format and check learning rates - no matter the input format this outputs learning rates in a standard format
        # where the learning rate starting at iteration 0 is guaranteed to be listed first
        learning_rate_its, learning_rate_values = format_and_check_learning_rates(learning_rates)

        # Determine what subjects we are fitting for
        if s_inds is None:
            n_subjects = len(self.s_collections)
            s_inds = range(n_subjects)
        n_fit_subjects = len(s_inds)

        # See how many penalizers we are working with
        if self.penalizers is None:
            n_penalizers = 0
        else:
            n_penalizers = len(self.penalizers)

        # Determine the devices the subject collections are on
        subject_coll_devices = [None]*n_fit_subjects
        for s_i, s_ind in enumerate(s_inds):
            subject_coll_devices[s_i] = next(self.s_collections[s_ind].s_mdl.parameters()).device

        n_smp_data_points = [len(self.s_collections[s_i].data) for s_i in s_inds]

        # Pull out groups of parameters with different learning rates
        base_parameters, _ = self.trainable_parameters(s_inds, get_prior_params=(fix_priors is False))
        if len(learning_rate_values[0]) > 1:  # Means learning rates specified for penalizers
            penalizer_params = [(self.get_penalizer_params(key), learning_rate_values[0][1][key])
                                for key in learning_rate_values[0][1].keys()]
        else:
            penalizer_params = [(self.get_penalizer_params(), learning_rate_values[0][-1])]

        # Setup initial optimizer
        params_with_lr = ([{'params': base_parameters, 'lr': learning_rate_values[0][0]}] +
                          [{'params': t[0], 'lr': t[1]} for t in penalizer_params])

        optimizer = torch.optim.Adam(params=params_with_lr, **adam_params)

        # Setup everything for checkpoints if we are creating them
        check_points = None
        if cp_epochs is not None:
            n_cps = len(cp_epochs)
            if n_cps > 0:
                cp_epochs = np.asarray(cp_epochs)
                check_points = [None]*n_cps

        # Setup everything for logging
        epoch_elapsed_time = np.zeros(n_epochs)
        epoch_nll = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_p_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_u_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_psi_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_scales_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_offsets_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_sub_direct_mappings_kl = np.zeros([n_epochs, n_fit_subjects])
        epoch_penalizer_penalties = np.zeros([n_epochs, n_penalizers])
        epoch_obj = np.zeros(n_epochs)

        # Perform fitting
        prev_learning_rates = learning_rate_values[0, :]
        for e_i in range(n_epochs):

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= e_i)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rates = learning_rate_values[cur_learing_rate_ind, :]
            if np.any(cur_learning_rates != prev_learning_rates):
                # We reset the whole optimizer because ADAM is an adaptive optimizer

                # Pull out groups of parameters with different learning rates
                if len(cur_learning_rates) > 1: # Means learning rates specified for penalizers
                    penalizer_params = [(self.get_penalizer_params(key), cur_learning_rates[1][key])
                                        for key in cur_learning_rates[1].keys()]
                else:
                    penalizer_params = [(self.get_penalizer_params(), cur_learning_rates[-1])]

                params_with_lr = ([{'params': base_parameters, 'lr': cur_learning_rates[0]}] +
                                  [{'params': t[0], 'lr': t[1]} for t in penalizer_params])

                optimizer = torch.optim.Adam(params=params_with_lr, **adam_params)
                prev_learning_rates = cur_learning_rates

            # Setup the iterators to go through the current data for this epoch in a random order
            epoch_batch_smp_inds = self.generate_batch_smp_inds(n_batches=n_batches, s_inds=s_inds)

            # Process each batch
            for b_i in range(n_batches):

                batch_obj_log = 0
                # Zero gradients
                optimizer.zero_grad()

                batch_nll = np.zeros(n_fit_subjects)
                batch_sub_p_kl = np.zeros(n_fit_subjects)
                batch_sub_u_kl = np.zeros(n_fit_subjects)
                batch_sub_psi_kl = np.zeros(n_fit_subjects)
                batch_sub_scales_kl = np.zeros(n_fit_subjects)
                batch_sub_offsets_kl = np.zeros(n_fit_subjects)
                batch_sub_direct_mappings_kl = np.zeros(n_fit_subjects)
                batch_penalizer_penalties = np.zeros(n_penalizers)
                for i, s_i in enumerate(s_inds):

                    s_coll = self.s_collections[s_i]

                    # Get the data for this batch for this subject, using efficient indexing if all data is
                    # already on the devices where the subject models are
                    batch_inds = epoch_batch_smp_inds[i][b_i]
                    if self.s_collections[s_i].data.data[0].device == subject_coll_devices[i]:
                        batch_data = self.s_collections[s_i].data.efficient_get_item(batch_inds)
                    else:
                        batch_data = self.s_collections[s_i].data[batch_inds]

                    # Send the data to the GPU if needed
                    batch_data.to(device=subject_coll_devices[i],
                                  non_blocking=subject_coll_devices[i].type == 'cuda')

                    # Form x and y for the batch
                    batch_x = [batch_data.data[i_g][batch_data.i_x, :] for i_g in s_coll.input_grps]
                    batch_y = [batch_data.data[i_h][batch_data.i_y, :] for i_h in s_coll.output_grps]
                    n_batch_data_pts = batch_x[0].shape[0]

                    # Make sure the input modules are on the right device for this subject
                    self.input_modules.to(subject_coll_devices[i])

                    # Make sure the posteriors are on the right GPU for this subject (important if we are
                    # using a shared posterior)
                    self._move_dists(s_coll.p_dists, subject_coll_devices[i])
                    self._move_dists(s_coll.u_dists, subject_coll_devices[i])
                    self._move_dists(s_coll.psi_dists, subject_coll_devices[i])
                    self._move_dists(s_coll.scale_dists, subject_coll_devices[i])
                    self._move_dists(s_coll.offset_dists, subject_coll_devices[i])
                    self._move_dists(s_coll.direct_mapping_dists, subject_coll_devices[i])

                    # Sample posteriors - note the helper function sets samples to None if there is no
                    # distribution over a parameter.  In this case, when we provide these None values
                    # to cond_forward() of the subject model, we indicate that the parameters in
                    # the subject model should be used.
                    q_p_modes, q_p_modes_standard = self._sample_posteriors(s_coll.p_dists, s_coll.props,
                                                                            s_coll.p_props)
                    q_u_modes, q_u_modes_standard = self._sample_posteriors(s_coll.u_dists, s_coll.props,
                                                                            s_coll.u_props)
                    q_psi_vls, q_psi_vls_standard = self._sample_posteriors(s_coll.psi_dists, s_coll.props,
                                                                            s_coll.psi_props, squeeze_std_smp=True)
                    q_scale_vls, q_scale_vls_standard = self._sample_posteriors(s_coll.scale_dists, s_coll.props,
                                                                                s_coll.scale_props,
                                                                                squeeze_std_smp=True)
                    q_offset_vls, q_offset_vls_standard = self._sample_posteriors(s_coll.offset_dists, s_coll.props,
                                                                                  s_coll.offset_props,
                                                                                  squeeze_std_smp=True)

                    q_direct_mapping_vls, q_direct_mapping_vls_standard = self._sample_posteriors(
                            s_coll.direct_mapping_dists, s_coll.props, s_coll.direct_mapping_props,
                            squeeze_std_smp=True)

                    # Make sure the m module is on the correct device for this subject, this is
                    # important when subject models share an m function
                    s_coll.s_mdl.m.to(subject_coll_devices[i])

                    # Apply input modules
                    batch_x = [x_g if i_m is None else i_m(x_g) for i_m, x_g in zip(self.input_modules, batch_x)]

                    # Calculate the conditional log-likelihood for this subject
                    y_pred = s_coll.s_mdl.cond_forward(x=batch_x, p=q_p_modes_standard, u=q_u_modes_standard,
                                                       scales=q_scale_vls_standard, offsets=q_offset_vls_standard,
                                                       direct_mappings=q_direct_mapping_vls_standard)
                    nll = (float(n_smp_data_points[i])/n_batch_data_pts)*s_coll.s_mdl.neg_ll(y=batch_y, mn=y_pred,
                                                                                             psi=q_psi_vls_standard)
                    nll.backward(retain_graph=True)
                    batch_obj_log += nll.detach().cpu().numpy()
                    batch_nll[i] = nll.detach().cpu().numpy()

                    # Calculate KL divergences between posteriors and priors
                    if enforce_priors:
                        s_p_kl_log = self._calc_kl_and_backward(dists0=s_coll.p_dists,
                                                                dists1=self.prior_collection.p_dists,
                                                                props=s_coll.props, prop_inds=s_coll.p_props,
                                                                smps=q_p_modes)
                        batch_sub_p_kl[i] = s_p_kl_log

                        s_u_kl_log = self._calc_kl_and_backward(dists0=s_coll.u_dists,
                                                                dists1=self.prior_collection.u_dists,
                                                                props=s_coll.props, prop_inds=s_coll.u_props,
                                                                smps=q_u_modes)
                        batch_obj_log += s_u_kl_log
                        batch_sub_u_kl[i] = s_u_kl_log

                        s_psi_kl_log = self._calc_kl_and_backward(dists0=s_coll.psi_dists,
                                                                  dists1=self.prior_collection.psi_dists,
                                                                  props=s_coll.props, prop_inds=s_coll.psi_props,
                                                                  smps=q_psi_vls)
                        batch_obj_log += s_psi_kl_log
                        batch_sub_psi_kl[i] = s_psi_kl_log

                        s_scales_kl_log = self._calc_kl_and_backward(dists0=s_coll.scale_dists,
                                                                     dists1=self.prior_collection.scale_dists,
                                                                     props=s_coll.props, prop_inds=s_coll.scale_props,
                                                                     smps=q_scale_vls)
                        batch_obj_log += s_scales_kl_log
                        batch_sub_scales_kl[i] = s_scales_kl_log

                        s_offsets_kl_log = self._calc_kl_and_backward(dists0=s_coll.offset_dists,
                                                                      dists1=self.prior_collection.offset_dists,
                                                                      props=s_coll.props, prop_inds=s_coll.offset_props,
                                                                      smps=q_offset_vls)
                        batch_obj_log += s_offsets_kl_log
                        batch_sub_offsets_kl[i] = s_offsets_kl_log

                        s_direct_mappings_kl_log = self._calc_kl_and_backward(dists0=s_coll.direct_mapping_dists,
                                                                    dists1=self.prior_collection.direct_mapping_dists,
                                                                    props=s_coll.props,
                                                                    prop_inds=s_coll.direct_mapping_props,
                                                                    smps=q_offset_vls)
                        batch_obj_log += s_direct_mappings_kl_log
                        batch_sub_direct_mappings_kl[i] = s_direct_mappings_kl_log

                # Apply penalizers
                for pen_i, penalizer in enumerate(self.penalizers):
                    batch_penalizer_penalties[pen_i] = penalizer.penalize_and_backwards()

                # Take a gradient step
                optimizer.step()

                # Make sure no private variance values are too small if we are fitting variances with point estimates
                with torch.no_grad():
                    for s_j in s_inds:
                        s_coll = self.s_collections[s_j]
                        s_min_var = s_coll.min_var
                        s_mdl = s_coll.s_mdl
                        for h in range(s_mdl.n_output_groups):
                            if s_coll.psi_dists[h] is None:
                                small_psi_inds = torch.nonzero(s_mdl.psi[h] < s_min_var[h])
                                s_mdl.psi[h].data[small_psi_inds] = s_min_var[h]

            # Handle checkpoints if needed
            if cp_epochs is not None:
                if np.any(cp_epochs == e_i):
                    # Clear the batch data from memory - this is helpful when working with GPU
                    del batch_x
                    del batch_y
                    del batch_data

                    print('Creating checkpoint after epoch ' + str(e_i) + '.')
                    cp_ind = np.argwhere(cp_epochs == e_i)[0][0]
                    check_points[cp_ind] = self.create_check_point(inc_penalizers=cp_penalizers)
                    check_points[cp_ind]['epoch'] = e_i

            # Take care of logging everything
            elapsed_time = time.time() - t_start
            epoch_elapsed_time[e_i] = elapsed_time
            epoch_nll[e_i, :] = batch_nll
            epoch_sub_p_kl[e_i, :] = batch_sub_p_kl
            epoch_sub_u_kl[e_i, :] = batch_sub_u_kl
            epoch_sub_psi_kl[e_i, :] = batch_sub_psi_kl
            epoch_sub_scales_kl[e_i, :] = batch_sub_scales_kl
            epoch_sub_offsets_kl[e_i, :] = batch_sub_offsets_kl
            epoch_sub_direct_mappings_kl[e_i, :] = batch_sub_direct_mappings_kl
            epoch_penalizer_penalties[e_i, :] = batch_penalizer_penalties
            epoch_obj[e_i] = batch_obj_log

            if e_i % update_int == 0:
                self._print_fitting_status(e_i=e_i, elapsed_time=elapsed_time, batch_obj=batch_obj_log,
                                           cur_learning_rates=cur_learning_rates, s_inds=s_inds, batch_nll=batch_nll,
                                           batch_sub_p_kl=batch_sub_p_kl, batch_sub_u_kl=batch_sub_u_kl,
                                           batch_sub_psi_kl=batch_sub_psi_kl, batch_sub_scales_kl=batch_sub_scales_kl,
                                           batch_sub_offsets_kl=batch_sub_offsets_kl,
                                           batch_sub_direct_mappings_kl=batch_sub_direct_mappings_kl,
                                           batch_penalties=batch_penalizer_penalties,
                                           devices=all_devices, penalizers=self.penalizers, print_opts=print_opts)

        # Return logs and check points
        if check_points is not None:
            check_points = [cp for cp in check_points if cp is not None]

        log = {'elapsed_time': epoch_elapsed_time, 'obj': epoch_obj, 'mdl_nll': epoch_nll, 'sub_p_kl': epoch_sub_p_kl,
               'sub_u_kl': epoch_sub_u_kl, 'sub_psi_kl': epoch_sub_psi_kl, 'sub_scales_kl': epoch_sub_scales_kl,
               'sub_offsets_kl': epoch_sub_offsets_kl, 'sub_direct_mappings_kl': epoch_sub_direct_mappings_kl,
               'penalties': epoch_penalizer_penalties}
        return [log, check_points]

    @classmethod
    def plot_log(cls, log: dict, show_obj: bool = True, show_mdl_nll: bool = True, show_p_kl: bool = True,
                 show_u_kl: bool = True, show_psi_kl: bool = True, show_scales_kl: bool = True,
                 show_offsets_kl: bool = True, show_direct_mappings_kl: bool = True, show_penalties: bool = True):
        """ Produces a figure of the values in a log produced by fit().

        Args:
            log: The log to plot.
            
            show_obj: True if the objective should be plotted through time
            
            show_mdl_nll: True if model negative log likelihood should be plotted through time
            
            show_p_kl: True if the kl divergences between prior and posterior distributions for p modes should be
                       plotted through time.
                         
            show_u_kl: True if the kl divergences between prior and posterior distributions for u modes should be
                       plotted through time. 
            
            show_psi_kl: True if the kl divergences between prior and posterior distributions for psi parameters should
                         be plotted through time. 
            
            show_scales_kl: True if the kl divergences between prior and posterior distributions for scale parameters 
                            should be plotted through time. 
            
            show_offsets_kl: True if the kl divergences between prior and posterior distributions for offset parameters 
                             should be plotted through time.

            show_direct_mappings_kl: True if the kl divergences between prior and posterior distributions for 
                                     direct mapping parameters should be plotted through time. 
             
        """

        n_plots = (show_obj + show_mdl_nll + show_p_kl + show_u_kl + show_psi_kl + show_scales_kl + show_offsets_kl +
                   show_direct_mappings_kl + show_penalties)

        n_rows = np.ceil(n_plots/2).astype('int')

        plt.figure()

        cnt = 0
        if show_obj:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['obj'])
            plt.title('Objective')

        if show_mdl_nll:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['mdl_nll'])
            plt.title('Model Negative Log Likelihoods')

        if show_p_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_p_kl'])
            plt.title('Subject P KL')

        if show_u_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_u_kl'])
            plt.title('Subject U KL')

        if show_psi_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_psi_kl'])
            plt.title('Subject Psi KL')

        if show_scales_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_scales_kl'])
            plt.title('Subject Scales KL')

        if show_offsets_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_offsets_kl'])
            plt.title('Subject Offsets KL')

        if show_direct_mappings_kl:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['sub_direct_mappings_kl'])
            plt.title('Subject DMs KL')

        if show_penalties:
            cnt += 1
            plt.subplot(n_rows, 2, cnt)
            plt.plot(log['elapsed_time'], log['penalties'])
            plt.title('Penalties')

    def to(self, device: torch.device, distribute_data: bool = False):
        """ Move everything in the fitter to a specified device.

        This is most useful when wanting to clean up after fitting and you need to move models to CPU.

        Args:
            device: The device to move everything to (e.g., torch.device('cpu'))

            distribute_data: True if data should be moved as well.

        """

        for s_coll in self.s_collections:
            s_coll.to(device, distribute_data=distribute_data)

        self.prior_collection.to(device)

        if self.penalizers is not None:
            for penalizer in self.penalizers:
                penalizer.to(device)

    @staticmethod
    def _calc_kl_and_backward(dists0: Sequence[Union[CondVAEDistribution, None]],
                          dists1: Sequence[Union[CondVAEDistribution, None]],
                          props: Sequence[torch.Tensor],
                          prop_inds: Union[Union[Sequence[int], None], None],
                          smps: Sequence[Union[torch.Tensor, None]]):
        """ Helper function for computing KL divergence between posterior and prior distributions and calling backwards.

        This functions handles the case when distributions and/or properties are not provided.

        Args:

            dists0: Posterior distributions.  If a None value is provided in the list, this serves as a placeholder
            indicating no calculations should be done.

            dist1: Prior distributions.

            props: List of properties

            prop_inds: props_inds[i] is the index in props that dists0[i] and dists[1] i should be conditioned on.

            smps: smps[i] is a set of samples to use for computing kl divergence.  Can be None if the kl calculation
            for a distribution type is known analytically.

        Returns:

            kl: The computed kl divergence

        """

        # If we have no distributions to worry about, this is easy
        if dists0 is None:
            return 0.0
        if all(v is None for v in dists0):
            return 0.0

        # If we have at least one distribution, then we compute kl divergences
        kl = 0.0
        for d0, d1, prop_ind, smp in zip(dists0, dists1, prop_inds, smps):
            if d0 is not None:
                if prop_ind is not None:
                    dist_props = props[prop_ind]
                else:
                    dist_props = None

                kl += torch.sum(d0.kl(d_2=d1, x=dist_props, smp=smp))

        kl.backward()
        return kl.detach().cpu().numpy()

    @staticmethod
    def _move_dists(dists: List, device: torch.device):
        """ Moves distributions to a device, avoiding calling .to() on None entries in a list. """
        if dists is not None:
            for d in dists:
                if d is not None:
                    d.to(device)

    @staticmethod
    def _print_fitting_status(e_i: int, elapsed_time: float, batch_obj: float, cur_learning_rates,
                              s_inds: Sequence[int], batch_nll: Sequence[float], batch_sub_p_kl: Sequence[float],
                              batch_sub_u_kl: Sequence[float], batch_sub_psi_kl: Sequence[float],
                              batch_sub_scales_kl: Sequence[float], batch_sub_offsets_kl: Sequence[float],
                              batch_sub_direct_mappings_kl: Sequence[float], batch_penalties: Sequence[float],
                              penalizers: Sequence[ParameterPenalizer], devices: List[torch.device],
                              print_opts: dict):
        """ Helper function for printing updates on fitting process.

        Args:
            e_i: The epoch index

            elapsed_time: Elapsed fitting time

            batch_obj: The objective value for the a batch of data (This will typically be the last batch of data
            for the epoch)

            cur_learning_rates: The current learning rates

            s_inds: The indices of the subjects that are being fit

            batch_nll: The negative log-likelihoods for the batch for each fit subject model
            (should correspond to s_inds)

            batch_sub_p_kl: The kl divergences for the p-mode distributions for each subject for the batch.

            batch_sub_u_kl: The kl divergences for the u-mode distributions for each subject for the batch.

            batch_sub_psi_kl: The kl divergences for the psi parameter distributions for each subject for the batch.

            batch_sub_scales_kl: The kl divergences for the scale parameter distributions for each subject for the
            batch.

            batch_sub_offsets_kl: The kl divergences for the offset parameter distributions for each subject for the
            batch.

            batch_sub_direct_mappings_kl: The kl divergences for the direct mapping parameter distributions for each
            subject for the batch.

            batch_penalties: The penalties each penalizer produced for the batch.

            penalizers: The penalizers used in fitting.

            devices: List of devices to print memory stats for.

            print_opts: A dictionary with fields with boolean values indicating which information should be shown.
            The fields are: mdl_nll, sub_kls, penalties, memory_usage.  Any fields that are not provided will be
            assumed to be false.
        """

        n_penalizers = len(batch_penalties)

        def _format_print_opts(d, f):
            if not (f in d.keys()):
                d[f] = False

        _format_print_opts(print_opts, 'mdl_nll')
        _format_print_opts(print_opts, 'sub_kls')
        _format_print_opts(print_opts, 'memory_usage')
        _format_print_opts(print_opts, 'penalties')
        _format_print_opts(print_opts, 'penalizer_states')

        print('*****************************************************')
        print('Epoch ' + str(e_i) + ' complete.  Obj: ' +
              '{:.2e}'.format(float(batch_obj)) +
              ', LR: ' + str(cur_learning_rates))
        if print_opts['mdl_nll']:
            print(format_output_list(base_str='Model NLLs: ', it_str='s_', vls=batch_nll, inds=s_inds))
        if print_opts['sub_kls']:
            print(format_output_list(base_str='Subj P KLs: ', it_str='s_', vls=batch_sub_p_kl, inds=s_inds))
            print(format_output_list(base_str='Subj U KLs: ', it_str='s_', vls=batch_sub_u_kl, inds=s_inds))
            print(format_output_list(base_str='Subj Psi KLs: ', it_str='s_', vls=batch_sub_psi_kl, inds=s_inds))
            print(format_output_list(base_str='Subj Scale KLs: ', it_str='s_', vls=batch_sub_scales_kl, inds=s_inds))
            print(format_output_list(base_str='Subj Offsets KLs: ', it_str='s_', vls=batch_sub_offsets_kl, inds=s_inds))
            print(format_output_list(base_str='Subj Direct Mappings KLs: ', it_str='s_',
                                     vls=batch_sub_direct_mappings_kl, inds=s_inds))
        if print_opts['penalties']:
            print(format_output_list(base_str='Penalties: ', it_str='p_', vls=batch_penalties,
                                     inds=np.arange(n_penalizers)))
        if print_opts['penalizer_states']:
            for penalizer in penalizers:
                print(penalizer)

        if print_opts['memory_usage']:
            device_memory_allocated = torch_devices_memory_usage(devices, type='memory_allocated')
            device_max_memory_allocated = torch_devices_memory_usage(devices, type='max_memory_allocated')
            print(format_output_list(base_str='Device memory allocated: ', it_str='d_',
                                     vls=device_memory_allocated, inds=range(len(device_memory_allocated))))
            print(format_output_list(base_str='Device max memory allocated: ', it_str='d_',
                                     vls=device_max_memory_allocated, inds=range(len(device_max_memory_allocated))))

        print('Elapsed time: ' + str(elapsed_time))

    @staticmethod
    def _sample_posteriors(dists: List[CondVAEDistribution], all_props: List[torch.Tensor], prop_inds: List[int],
                           squeeze_std_smp: bool = False):
        """ Samples posteriors, returning samples in compact and standard form.

        Args:
            dists: List of distributions to sample.  Entries can be None (allowing for placeholders)

            all_props: List of property tensors.

            prop_inds: prop_inds[i] contains is the index in all_props for dists[i].  If prop_inds[i] is None,
            then sampling is done without conditioning on properties.

        Returns:

            smps: smps[i] is the compact samples for dists[i].  If dists[i] was none, smps[i] will ne none.

            std_smps: std_smps[i] is smps[i] in standard form.  If dists[i] was none, std_smps[i] will be none.

            squeeze_std_sample: True if standard sample should be squeezed before returning.
        """
        if dists is None:
            return None, None

        n_dists = len(dists)
        if prop_inds is None:
            prop_inds = [None]*n_dists

        smps = [None]*n_dists
        std_smps = [None]*n_dists

        for d_i, d in enumerate(dists):
            if d is not None:
                if prop_inds[d_i] is not None:
                    smps[d_i] = d.sample(all_props[prop_inds[d_i]])
                else:
                    smps[d_i] = d.sample()
                std_smps[d_i] = d.form_standard_sample(smps[d_i])

        if squeeze_std_smp:
            std_smps = [torch.squeeze(s) if s is not None else None for s in std_smps]

        return smps, std_smps


def eval_fits(s_collections: Sequence[SubjectVICollection], data: TimeSeriesBatch, batch_size: int = 100,
             metric: Callable = None, return_preds: bool = True) -> List:
    """ Measures model fits on a given set of data.

    This function generates predictions for each model using the posterior means of modes. It then evaluates these
    predictions using negative log-likelihood by default but the user can specify other metrics for measuring
    prediction quality.

    Args:
        s_collections: A sequence of VI collections we want to evaluate.

        data: The data to use for evaluation

        batch_size: The number of samples to send to GPU at one time for evaluation; can be useful if working with
        low-memory GPUs.

        metric: A function which computes fit quality given the output of predict_from_truth

        return_preds: True if predictions should be returned

    Returns:
        metrics: metrics[i] is the fit quality for s_collections[i].  Note that if no metric is supplied, this will
        just be a list of negative log-likelihood values, but custom metric functions can return arbitrary objects.

        preds_with_truth: preds_with_truth[i] are the predictions for s_collections[i] produced with the function
        predict_with_turth. This will only be returned in return_preds is true.
    """

    # Generate predictions
    n_mdls = len(s_collections)
    preds_with_truth = [None]*n_mdls
    metrics = [None]*n_mdls
    for c_i, s_coll_i in enumerate(s_collections):
        if c_i % 10 == 0:
            print('Generating predictions for fit: ' + str(c_i))

        # Make prediction
        pred_i = predict_with_truth(s_collection=s_coll_i, data=data, batch_size=batch_size, time_grp=None)

        # Evaluate fits
        if metric is None:
            y = [torch.Tensor(pred_i['truth'][h]) for h in range(len(pred_i['truth']))]
            mn = [torch.Tensor(pred_i['pred'][h]) for h in range(len(pred_i['pred']))]
            with torch.no_grad():
                metrics[c_i] = s_coll_i.s_mdl.neg_ll(y=y, mn=mn).detach().cpu().numpy().item()
        else:
                metrics[c_i] = metric(pred_i)

        if return_preds:
            preds_with_truth[c_i] = pred_i

    if return_preds:
        return [metrics, preds_with_truth]
    else:
        return metrics


def predict(s_collection: SubjectVICollection, data: TimeSeriesBatch, batch_size: int = 100) -> List[np.ndarray]:
    """ Predicts output given input from a model with posterior distributions over parameters.

    If posterior distributions for a paremeter are present, predictions are based on the posterior mean.  If
    posterior distributions for a parameter are not present, the parameter value in the subject model
    will be used.

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

    # Define a helper function
    def _get_post_means(dists: List[CondVAEDistribution], all_props: List[torch.Tensor], prop_inds: List[int],
                        squeeze_output: bool = False):
        """ Gets the posterior means for a list of distributions, handling cases where some distributions are None.

        Args:
            dists: The distributions to get posterior means for.  Can be None, in which case this function returns None.

            all_props: List of properties.

            prop_inds: prop_inds[i] is the index into all_props for the properties for dists[i].  If no properties
            are needed for the distribution prop_inds[i] should be None.  If no properties are needed for all
            of the distributions, prop_inds can be None.

            squeeze_output: True if output should be squeezed.

        Returns:

            mns: The means of distributions.  mns[i] is the mean for distribution i and will be None if dists[i] was
            None.  mns will be None if dists was None.
        """

        if dists is None:
            return None

        n_dists = len(dists)
        if prop_inds is None:
            prop_inds = [None]*n_dists

        mns = [None]*n_dists
        for d_i, d in enumerate(dists):
            if d is not None:
                if prop_inds[d_i] is not None:
                    mns[d_i] = d(all_props[prop_inds[d_i]])
                else:
                    mns[d_i] = d()

        if squeeze_output:
            mns = [torch.squeeze(mn) if mn is not None else None for mn in mns]

        return mns

    # Get the parameters

    p = _get_post_means(s_collection.p_dists, s_collection.props, s_collection.p_props)
    u = _get_post_means(s_collection.u_dists, s_collection.props, s_collection.u_props)
    scales = _get_post_means(s_collection.scale_dists, s_collection.props, s_collection.scale_props,
                             squeeze_output=True)
    offsets = _get_post_means(s_collection.offset_dists, s_collection.props, s_collection.offset_props,
                              squeeze_output=True)
    direct_mappings = _get_post_means(s_collection.direct_mapping_dists, s_collection.props,
                                      s_collection.direct_mapping_props, squeeze_output=True)

    s_collection_device = next(s_collection.s_mdl.parameters()).device

    y = [None]*n_batches
    batch_start = 0
    for b_i in range(n_batches):
        batch_end = batch_start + batch_size
        batch_data = data[batch_start:batch_end]

        # Move data to device the collection is on
        batch_data.to(s_collection_device, non_blocking=s_collection_device.type == 'cuda')

        # Form x
        batch_x = [batch_data.data[i_g][batch_data.i_x] for i_g in s_collection.input_grps]
        with torch.no_grad():
            batch_y = s_collection.s_mdl.cond_forward(x=batch_x, p=p, u=u, scales=scales, offsets=offsets,
                                                      direct_mappings=direct_mappings)
        batch_y = [t.cpu().numpy() for t in batch_y]
        y[b_i] = batch_y

        batch_start = batch_end

    # Concatenate output
    n_output_grps = len(y[0])
    y_out = [None]*n_output_grps
    for h in range(n_output_grps):
        y_out[h] = np.concatenate([batch_y[h] for batch_y in y])

    return y_out


def predict_with_truth(s_collection: SubjectVICollection, data: TimeSeriesBatch, batch_size: int = 100,
                       time_grp: int = None):
    """ Predicts output for a model, using posterior over modes, and including true data in output for reference.

    This is a wrapper function around predict for convenience.

    Note: All predictions will be returned on host (cpu) memory as numpy arrays.

    Args:
        s_collection: The collection for the subject.   Any data in the collection will be ignored.

        data: The data to predict with.

        batch_size: The number of samples we predict on at a time.  This is helpful if using a GPU
        with limited memory.

        time_grp: The index of the group in data with time stamps.  If None, no time stamps will be returned.

    Returns:

        predictions: A dictionary with the following keys:
            pred: predictions. pred[h] is the prediction for the h^th output group of the model
            truth: Corresponding true values for those in pred
            time: It time_grp is not None, the time indices for each point in pred and truth. Otherwise, time will be
            None.

    """

    output_grps = s_collection.output_grps

    pred = predict(s_collection=s_collection, data=data, batch_size=batch_size)
    truth = [data.data[h][data.i_y].cpu().numpy() for h in output_grps]

    if time_grp is not None:
        time = data.data[time_grp][data.i_y].cpu().numpy()
    else:
        time = None

    return {'pred': pred, 'truth': truth, 'time': time}


