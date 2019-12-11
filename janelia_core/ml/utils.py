""" Contains basic machine learning utility functions.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

import numpy as np
import torch




def format_and_check_learning_rates(learning_rates):
    """ Takes learning rates in different formats and puts them in a standard format, running basic checks.

    In iterative optimization routines, users may provide a single learning rate for all iterations or may provide a
    schedule of learning rates.  This function accepts input as either a single number, specifying the learning rate to
    use on all iterations, or as a list of tuples (specifying a learning rate schedule, see below) and outputs, in all
    cases, a learning rate schedule which is checked for consistency and in a standard format.

    Args:
        learning_rates: If a single number, this is the learning rate to use for all iteration.  Alternatively, this
            can be a list of tuples.  Each tuple is of the form (iteration, learning_rate), which gives the learning rate
            to use from that iteration onwards, until another tuple specifies another learning rate to use at a different
            iteration on.  E.g., learning_rates = [(0, .01), (1000, .001), (10000, .0001)] would specify a learning
            rate of .01 from iteration 0 to 999, .001 from iteration 1000 to 9999 and .0001 from iteration 10000 onwards.

    Returns:
        learning_rate_its: A sorted numpy array of the iterations each learning rate comes into effect on.
        learning_rate_values: A numpy array of the corresponding learning rates for learning_rate_its.  Specifically,
            learning_rate_values[i] comes into effect on iteration learning_rate_its[i]

    Raises:
        ValueError: If learning_rates specifies two or more learning rates for the same iteration
        ValueError: If learning rates are specified as a list of tuples and negative iterations are supplied
        ValueError: If learning rates are specified as a list of tuples and iteration

    """

    if not isinstance(learning_rates, (int, float, list)):
        raise (ValueError('learning_rates must be of type int, float or list.'))

    if isinstance(learning_rates, (int, float)):
        learning_rates = [(0, learning_rates)]

    learning_rate_its = np.asarray([t[0] for t in learning_rates])
    learning_rate_values = np.asarray([t[1] for t in learning_rates])

    if len(learning_rate_its) != len(np.unique(learning_rate_its)):
        raise(ValueError('Two or more learning rates specified for the same iteration.'))
    if any(learning_rate_its < 0):
        raise(ValueError('Learning rates specified for one or more negative iterations.'))
    if not any(learning_rate_its == 0):
        raise(ValueError('Learning rates must specify learning rates starting at iteration 0'))

    sort_order = np.argsort(learning_rate_its)
    learning_rate_its = learning_rate_its[sort_order]
    learning_rate_values = learning_rate_values[sort_order]

    return [learning_rate_its, learning_rate_values]


def torch_mod_to_fcn(m: torch.nn.Module):
    """ Converts a torch Module to a standard python function.

    Returns a new python function which:
        1) Takes numpy input and converts that input to a torch Tensor
        2) Calls the forward method of the torch module on that input (in the context of no_grad)
        3) Converts the output to a numpy array
    """

    def wrapper_fcn(x):
        x_t = torch.Tensor(x)
        with torch.no_grad():
            y_t = m(x_t)
        return y_t.numpy()

    return wrapper_fcn


def list_torch_devices(verbose: bool = True):
    """ Returns a list of torch devices.  Will return *either* cpu or a list of GPUs.

    If GPUs are available, will return a list of GPUs over returning the cpu. Will only
    return cpu when no GPUs are available.

    Args:
        verbose: True if summary of found devices should be printed to screen.

    Returns:
        devices: List of devices

        cuda_is_available: True if returned devices are GPU
    """

    if torch.cuda.is_available():
        n_cuda_devices = torch.cuda.device_count()
        if verbose:
            print('Found ' + str(n_cuda_devices) + ' GPUs')
        devices = [torch.device('cuda:' + str(i)) for i in range(n_cuda_devices)]
        cuda_is_available = True
    else:
        if verbose:
            print('No GPUs found.')
        devices = [torch.device('cpu')]
        cuda_is_available = False

    return [devices, cuda_is_available]


def torch_devices_memory_usage(devices: Sequence[torch.device], type: str = 'max_memory_allocated')-> list:
    """ Returns a list of memory usage on devices.

    This function gets either memory allocated or max memory on torch devices, providing a value of nan if a device is
    a cpu.

    Args:
        devices: List of devices

        type: 'max_memory_allocated' to list max memory allocated; for any other string memory allocated is returned

    Returns:
        mem_usage: The list of memory usage
    """
    if str == 'max_memory_allocated':
        return [torch.cuda.max_memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in devices]
    else:
        return [torch.cuda.memory_allocated(device=d) if d.type == 'cuda' else np.nan for d in devices]
