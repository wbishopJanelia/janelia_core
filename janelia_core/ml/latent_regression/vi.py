""" Tools for fitting latent regression models with variational inference. """

import itertools
import time
from typing import Sequence

import numpy as np
import torch

from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.torch_distributions import CondVAEDistriubtion
from janelia_core.ml.utils import format_and_check_learning_rates

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
        log: A dictionary logging progress.  Will have the enries:
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