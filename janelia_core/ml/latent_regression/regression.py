""" Tools for working with latent regression models where we use point estimates of modes instead of distributions. """

import itertools
import time
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from janelia_core.ml.datasets import TimeSeriesBatch
from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.utils import format_and_check_learning_rates
from janelia_core.ml.utils import torch_devices_memory_usage

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


class SubjectRegressionCollection:
    """ An object which holds subject data, properties and likelihood model.

    """

    def __init__(self, s_mdl: LatentRegModel, data: TimeSeriesBatch,
                 input_grps: Sequence, output_grps: Sequence,
                 props: Sequence[torch.Tensor], input_props: Sequence[int],
                 output_props: Sequence[int],
                 p_preds: Sequence[Union[torch.nn.Module, torch.Tensor]],
                 u_preds: Sequence[Union[torch.nn.Module, torch.Tensor]]):
        """ Creates a new SubjectVICollection object.

        Args:
            s_mdl: The likelihood model for the subject.

            data: Data for the subject.

            input_grps: input_grps[g] is the index into data.data for the g^th input group

            output_grps: output_grps[h] is the index into data.data for the h^th output group

            props: props[i] is a tensor of properties for one or more input or output groups of variables

            input_props: input_props[g] is the index into props for the properties for the g^th input group.  If
            the modes for the g^th input group are fixed, then input_props[g] should be None.

            output_props: output_props[h[ is the index into props for the properties of the g^th output group.  If
            the modes for the h^th output group are fixed, then outpout_props[h] should be None.

            p_preds: p_pred[g] is the predictor to use for the modes of the g^th input group.  If the modes for this
            input group are fixed, then g should be a tensor with the fixed values.

            u_preds: u_preds are the predictors for the modes for the output groups, same format as p_preds.

        """

        self.s_mdl = s_mdl
        self.data = data
        self.input_grps = input_grps
        self.output_grps = output_grps
        self.props = props
        self.input_props = input_props
        self.output_props = output_props
        self.p_preds = p_preds
        self.u_preds = u_preds
        self.device = None # Initially we don't specify which device everything is on (allowing things to potentially
                           # be on multiple devices).

    def trainable_parameters(self) -> Sequence:
        """ Returns all trainable parameters for the collection.

        Returns:
            params: The list of parameters.
        """
        p_pred_params = itertools.chain(*[p.parameters() for p in self.p_preds if not isinstance(p, torch.Tensor)])
        u_pred_params = itertools.chain(*[p.parameters() for p in self.u_preds if not isinstance(p, torch.Tensor)])


        return list(itertools.chain(p_pred_params, u_pred_params, self.s_mdl.trainable_parameters()))


    def to(self, device: Union[torch.device, int], distribute_data: bool = False):
        """ Moves all relevant attributes of the collection to the specified device.

        This will move the subject model and properties to the device as well as predictors.

        Note that by default fitting data will not be moved to the device.

        Args:
            device: The device to move attributes to.

            distribute_data: True if fitting data should be moved to the device as well
        """

        self.s_mdl = self.s_mdl.to(device)
        self.props = [p.to(device) for p in self.props]
        self.p_preds = [p.to(device) for p in self.p_preds]
        self.u_preds = [p.to(device) for p in self.u_preds]

        self.device = device

        if distribute_data and self.data is not None:
            self.data.to(device)


class MultiSubjectRegressionFitter():
    """ Object for fitting a collection of latent regression models with point estimates for modes.

    """

    def __init__(self, s_collections: Sequence[SubjectRegressionCollection], min_var: float = .001):

        """ Creates a new MultiSubjectRegressionFitter object.

        Args:
            s_collections: A set of SubjectRegressionCollections to use when fitting data.

            min_var: The minimum variance to enforce on the final noise distributions of output variables.

        """

        self.s_collections = s_collections
        self.min_var = min_var

        self.distributed = False  # Keep track if we have distributed everything yet

    def distribute(self, devices: Sequence[Union[torch.device, int]], s_inds: Sequence[int] = None,
                   distribute_data: bool = False):
        """ Distributes subject regression collections across devices.

        Args:
            devices: Devices that things can be distributed across.

            s_inds: Indices into self.s_collections for subject regression collections which should be distributed
            across devices.  If None, collections for all subjects will be distributed.

            distribute_data: True if all training data should be distributed to devices.  If there is enough
            device memory, this can speed up fitting.  If not, set this to false, and batches of data will
            be sent to the device for each training iteration.

        """
        if s_inds is None:
            s_inds = range(len(self.s_collections))

        n_devices = len(devices)

        # Distribute subject collections
        for i, s_i in enumerate(s_inds):
            device_ind = (i+1) % n_devices
            self.s_collections[s_i].to(devices[device_ind], distribute_data=distribute_data)

        self.distributed = True

    def trainable_parameters(self, s_inds: Sequence[int] = None) -> list:
        """ Gets all trainable parameters for fitting predictors and a set of subjects.

        Args:
            s_inds: Specifies the indices of subjects that will be fit.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

        Returns:
             params: Parameters of mode predictors and subject models
        """

        collection_params = itertools.chain(*[self.s_collections[s_i].trainable_parameters() for s_i in s_inds])

        # Clean for duplicate parameters.  Subject models might shared a posterior, for example, so we need to
        # check for this
        non_duplicate_params = list(set(collection_params))

        return non_duplicate_params

    def generate_batch_smp_inds(self, n_batches: int, s_inds: Sequence[int] = None):
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
        """ Lists the devices the subject models and predictors are on.

        Returns:
            devices: The list of devices subject models and priors are on.
        """

        s_coll_devices = [s_coll.device for s_coll in self.s_collections]

        return list(set([*s_coll_devices]))

    def fit(self, n_epochs: int = 10, n_batches: int = 10, learning_rates = .01,
            adam_params: dict = {}, s_inds: Sequence[int] = None, update_int: int = 1,
            print_mdl_nlls: bool = True, print_memory_usage: bool = True) -> dict:
        """

        Args:

            n_epochs: The number of epochs to run fitting for.

            n_batches: The number of batches to break the training data up into per epoch.  When multiple subjects have
            different numbers of total training samples, the batch size for each subject will be selected so we go
            through the entire training set for each subject after processing n_batches each epoch.

            learning_rates: If a single number, this is the learning rate to use for all epochs and parameters.
            Alternatively, this can be a list of tuples.  Each tuple is of the form
            (epoch,lr), where epoch is the epoch the learning rates come into effect on, lr is the learning rate.
            Multiple tuples can be provided to give a schedule of learning rates.  Here is an example
            learning_rates: [(0, .001), (100, .0001)] that starts with a learning rates of .001, and epoch 100, the
            learning rate goes to .0001.

            adam_params: Dictionary of parameters to pass to the call when creating the Adam Optimizer object.
            Note that *if learning rate is specified here it will be ignored.* (Use the learning_rates option instead).
            The options specified here will be applied to all parameters at all iterations.

            s_inds: Specifies the indices of subjects to fit to.  Subject indices correspond to their
            original order in s_collections when the fitter was created. If None, all subjects used.

            update_int: Fitting status will be printed to screen every update_int number of epochs

            print_mdl_nlls: If true, when fitting status is printed to screen, the negative log likelihood of each
            evaluated model will be printed to screen.

            print_memory_usage: If true, when fitting status is printed to screen, the memory usage of each
            device will be printed to streen.

        Return:
            log: A dictionary with the following entries:

                'elapsed_time': A numpy array of the elapsed time for completion of each epoch

                'mdl_nll': mdl_nll[e, i] is the negative log likelihood for the subject model s_inds[i] at the start
                of epoch e (that is when the objective has been calculated but before parameters have been updated)

                obj: obj[e] contains the objective value at the start of epoch e.  This is the negative evidence lower
                bound + weight penalties.

        Raises:
            RuntimeError: If distribute() has not been called before fitting.

        """

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

        n_smp_data_points = [len(self.s_collections[s_i].data) for s_i in s_inds]

        # Pull out groups of parameters with different learning rates
        params  = self.trainable_parameters(s_inds)

        # Setup initial optimizer
        optimizer = torch.optim.Adam(params, lr=learning_rate_values[0,0], **adam_params)

        # Setup everything for logging
        epoch_elapsed_time = np.zeros(n_epochs)
        epoch_nll = np.zeros([n_epochs, n_fit_subjects])
        epoch_obj = np.zeros(n_epochs)

        # Perform fitting
        prev_learning_rate = learning_rate_values[0, 0]
        for e_i in range(n_epochs):

            # Set the learning rate
            cur_learing_rate_ind = np.nonzero(learning_rate_its <= e_i)[0]
            cur_learing_rate_ind = cur_learing_rate_ind[-1]
            cur_learning_rate = learning_rate_values[cur_learing_rate_ind, 0]
            if cur_learning_rate != prev_learning_rate:
                # We reset the whole optimizer because ADAM is an adaptive optimizer
                optimizer = torch.optim.Adam(params=params, lr=cur_learning_rate, **adam_params)
                prev_learning_rate = cur_learning_rate

            # Setup the iterators to go through the current data for this epoch in a random order
            epoch_batch_smp_inds = self.generate_batch_smp_inds(n_batches=n_batches, s_inds=s_inds)

            # Process each batch
            for b_i in range(n_batches):

                batch_obj_log = 0

                # Zero gradients
                optimizer.zero_grad()

                batch_nll = np.zeros(n_fit_subjects)
                for i, s_i in enumerate(s_inds):

                    s_coll = self.s_collections[s_i]

                    # Get the data for this batch for this subject, using efficient indexing if all data is
                    # already on the devices where the subject models are
                    batch_inds = epoch_batch_smp_inds[i][b_i]
                    if s_coll.data.data[0].device == s_coll.device:
                        batch_data = s_coll.data.efficient_get_item(batch_inds)
                    else:
                        batch_data = s_coll.data[batch_inds]

                        # Send the data to the GPU if needed
                        batch_data.to(device=s_coll.device, non_blocking=s_coll.device.type == 'cuda')

                    # Form x and y for the batch
                    batch_x = [batch_data.data[i_g][batch_data.i_x, :] for i_g in s_coll.input_grps]
                    batch_y = [batch_data.data[i_h][batch_data.i_y, :] for i_h in s_coll.output_grps]
                    n_batch_data_pts = batch_x[0].shape[0]

                    # Make sure all predictors are on the right GPU for this subject
                    for p in s_coll.p_preds:
                        p.to(s_coll.device)
                    for p in s_coll.u_preds:
                        p.to(s_coll.device)

                    # Generate mode predictions for this subject
                    p_modes = [p if isinstance(p, torch.Tensor)
                               else p(s_coll.props[s_coll.input_props[g]])
                               for g, p in enumerate(s_coll.p_preds)]

                    u_modes = [p if isinstance(p, torch.Tensor)
                               else p(s_coll.props[s_coll.output_props[h]])
                               for h, p in enumerate(s_coll.u_preds)]

                    # Make sure the m module is on the correct device for this subject, this is
                    # important when subject models share an m function
                    s_coll.s_mdl.m = s_coll.s_mdl.m.to(s_coll.device)

                    # Calculate the conditional log-likelihood for this subject
                    y_pred = s_coll.s_mdl.cond_forward(x=batch_x, p=p_modes, u=u_modes)
                    nll = (float(n_smp_data_points[i])/n_batch_data_pts)*s_coll.s_mdl.neg_ll(y=batch_y, mn=y_pred)
                    nll.backward(retain_graph=True)
                    batch_obj_log += nll.detach().cpu().numpy()

                    # Record the log likelihood  for each subject for logging
                    batch_nll[i] = nll.detach().cpu().numpy()

                # Take a gradient step
                optimizer.step()

                # Make sure no private variance values are too small
                with torch.no_grad():
                    for s_j in s_inds:
                        s_mdl = self.s_collections[s_j].s_mdl
                        for h in range(s_mdl.n_output_groups):
                            small_psi_inds = torch.nonzero(s_mdl.psi[h] < self.min_var)
                            s_mdl.psi[h].data[small_psi_inds] = self.min_var

            # Take care of logging everything
            elapsed_time = time.time() - t_start
            epoch_elapsed_time[e_i] = elapsed_time
            epoch_nll[e_i, :] = batch_nll
            epoch_obj[e_i] = batch_obj_log

            if e_i % update_int == 0:
                print('*****************************************************')
                print('Epoch ' + str(e_i) + ' complete.  Obj: ' +
                      '{:.2e}'.format(float(batch_obj_log)) +
                      ', LR: '  + str(cur_learning_rate ))

                if print_mdl_nlls:
                    print(format_output_list(base_str='Model NLLs: ', it_str='s_', vls=batch_nll, inds=s_inds))

                if print_memory_usage:
                    device_memory_allocated = torch_devices_memory_usage(all_devices, type='memory_allocated')
                    device_max_memory_allocated = torch_devices_memory_usage(all_devices, type='max_memory_allocated')
                    print(format_output_list(base_str='Device memory allocated: ', it_str='d_',
                          vls=device_memory_allocated, inds=range(len(device_memory_allocated))))
                    print(format_output_list(base_str='Device max memory allocated: ', it_str='d_',
                          vls=device_max_memory_allocated, inds=range(len(device_max_memory_allocated))))

                print('Elapsed time: ' + str(elapsed_time))

        # Return logs
        log = {'elapsed_time': epoch_elapsed_time, 'mdl_nll': epoch_nll, 'obj': epoch_obj}
        return log

    @classmethod
    def plot_log(cls, log: dict):
        """ Produces a figure of the values in a log produced by fit().

        Args:
            log: The log to plot.
        """
        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(log['elapsed_time'], log['obj'])
        plt.title('Objective')

        plt.subplot(2, 1, 2)
        plt.plot(log['elapsed_time'], log['mdl_nll'])
        plt.title('Model Negative Log Likelihoods')


def predict(s_collection: SubjectRegressionCollection, data: TimeSeriesBatch, batch_size: int = 100) -> List[np.ndarray]:
    """ Predicts output given input from a model with predictions over modes.

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

    # Make sure the predictors are on the correct device
    for p_pred in s_collection.p_preds:
        p_pred.to(s_collection.device)

    for u_pred in s_collection.u_preds:
        u_pred.to(s_collection.device)

    # Get the posterior means for the modes
    p_modes = [p if isinstance(p, torch.Tensor)
               else p(s_collection.props[s_collection.input_props[g]])
               for g, p in enumerate(s_collection.p_preds)]

    u_modes = [p if isinstance(p, torch.Tensor)
               else p(s_collection.props[s_collection.output_props[h]])
               for h, p in enumerate(s_collection.u_preds)]

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
            batch_y = s_collection.s_mdl.cond_forward(x=batch_x, p=p_modes, u=u_modes)
        batch_y = [t.cpu().numpy() for t in batch_y]
        y[b_i] = batch_y

        batch_start = batch_end

    # Concatenate output
    n_output_grps = len(y[0])
    y_out = [None]*n_output_grps
    for h in range(n_output_grps):
        y_out[h] = np.concatenate([batch_y[h] for batch_y in y])

    return y_out

