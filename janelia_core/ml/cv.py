""" Contains tools for performing cross validation with data.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np

from janelia_core.math.basic_functions import divide_into_nearly_equal_parts


def generate_balanced_folds(labels: np.ndarray, n_folds: int, balance_across_folds: bool = False,
                            balance_within_folds: bool = False, test_train_closure: bool = False,
                            randomize: bool = True) -> list:

    """ Given an array of labels, generate test and train folds with balanced labels.

    This function first breaks the data into disjoint testing sets.  By default, this function will
    assign points into disjoint test folds so that all points are used for testing once.  In this case,
    the number of data points of a class in any test fold will be within 1 of eachother. However, see
    the options to balance_within_folds and balance_across_folds for alternative behavior.

    After assigning test points for each fold, the training points for each fold are assigned. By default,
    all points not used for testing and used for training in a fold. However, see test_train_closure option
    for the different ways training points can be assigned to sets.

    Args:
        labels: labels[i] contains the label for data point i

        n_folds: The number of folds to break the data into

        balance_across_folds: If true, this ensures the number of test points of each class is the
        same across folds. (Though within folds, the number of test points of each class may differ).

        balance_within_folds: If true, this ensures the number of test points of each class is the
        same in each fold.   (Though across folds, the number of test points of each class may differ.)

        test_train_closure: If this is a true, the union across folds of test points must
        equal the union across folds of train points.  This ensures that if a point is used
        for training in one fold, it will also be used for testing in the other folds.

        randomize: True if assignment to folds should be randomized.

    Returns:
        folds: A list of length n_folds.  folds[i] contains a dictionary with the fields:
            test_inds: Containing indices of test points for the fold
            train_inds: Containing indices of train points for the fold

    Raises:
        RuntimeError: If there are not enough points of each class to have at least one point of each class
        in each test fold.
    """

    n_data_pts = len(labels)

    # See how many points of each class there are
    class_labels, class_counts = np.unique(labels, return_counts=True)

    # Make sure we have enough points of each class
    for idx, cnt in enumerate(class_counts):
        if cnt < n_folds:
            raise(RuntimeError('Not enough data points with class label ' + str(class_labels[idx])  +
                  ' to break into ' + str(n_folds) + ' folds.'))

    # Cut down number of data points to balance classes across folds if needed
    if balance_across_folds:
        class_counts = np.asarray([n_folds*(cnt//n_folds) for cnt in class_counts])

    # Cut down number of data points to balance classes within folds if needed
    if balance_within_folds:
        min_cnt = np.min(class_counts)
        class_counts = min_cnt*np.ones(class_counts.shape, np.int)

    # Assign test points for each fold
    folds = [{'test_inds': np.zeros(0, np.int), 'train_inds': np.zeros(0, np.int)} for f_i in range(n_folds)]

    for cls_i, cls_l in enumerate(class_labels):
        data_inds = np.where(labels == cls_l)[0]

        if randomize:
            data_inds = np.random.permutation(data_inds)

        n_test_smps_per_fold = divide_into_nearly_equal_parts(class_counts[cls_i], n_folds)

        start_i = 0
        for f_i in range(n_folds):
            end_i = start_i + n_test_smps_per_fold[f_i]
            folds[f_i]['test_inds'] = np.concatenate([folds[f_i]['test_inds'], data_inds[start_i:end_i]])
            start_i = end_i

    # Assign train points
    for f_i in range(n_folds):
        if test_train_closure:
            train_inds = [folds[c_i]['test_inds'] for c_i in range(n_folds) if c_i != f_i]
            train_inds = np.concatenate(train_inds)
            folds[f_i]['train_inds'] = train_inds
        else:
            folds[f_i]['train_inds'] = np.setdiff1d(range(n_data_pts), folds[f_i]['test_inds'])

    return folds


