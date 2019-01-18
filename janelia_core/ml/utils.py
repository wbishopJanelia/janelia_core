""" Contains basic machine learning utility functions.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


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
