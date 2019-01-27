""" Contains tools for performing linear discriminant analysis.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
import scipy.linalg


def fisher_lda(x: np.ndarray, y: np.ndarray, use_cov: bool = False) -> np.ndarray:
    """ Computes Fisher LDA weights.

    This function will calculate w = argmax w^TS_Bw / w^TS_Ww where:

        S_B is the between class scatter matrix S_B = \sum_c (mu_c - mu)(mu_c - mu)^T,
        where mu_c is the mean for class c and mu is mean over all data points.

        S_W = \sum_c S_w_c, where S_w_c = \sum_i (x_i - mu_c)(x_i - mu_c)^T, where the sum
        is over x_i belonging to class c.

        This function will return C - 1 w vector, where C is the number of classes.

    Args:
        x: An array with feature vectors.  Each row is a vector.

        y: An array with feature labels.

        use_cov: If true, covariance matrices are used for each class scatter matrix instead
        of  \sum_i (x_i - mu_c)(x_i - mu_c)^T.  This has a balancing affect if the number of
        data points is different for each class.

    Returns:

        w: An array with weight vectors. Each column is a weight vector.

    Example:

        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.random
        import scipy.linalg

        # Generate some data

        d_x = 10
        n_labels_per_class = [100, 100, 10]

        n_classes = len(n_labels_per_class)
        class_means = list()
        x = list()
        y = list()
        for c_i in range(n_classes):
            class_mean = np.random.randn(d_x)
            class_x = np.random.multivariate_normal(class_mean, np.eye(d_x), n_labels_per_class[c_i])
            class_y = c_i*np.ones(n_labels_per_class[c_i])

            class_means.append(class_mean)
            x.append(class_x)
            y.append(class_y)

        x = np.concatenate(x)
        y = np.concatenate(y)

        # Calcualte fisher LDA weights
        w = fisher_lda(x, y, use_cov=True)

        ## Project the data
        x_p = np.matmul(x, w)

        ## Plot the data
        for c_i in range(n_classes):
            c_inds = np.where(y == c_i)[0]
            plt.plot(x_p[c_inds,0], x_p[c_inds,1], 'o')


    """

    d_x = x.shape[1]

    class_lbls = np.unique(y)
    n_classes = len(class_lbls)

    # Calculate within and between scatter matrices

    mu = np.mean(x, 0)

    s_b = np.zeros([d_x, d_x])
    s_w = np.zeros([d_x, d_x])

    for c_i, c_lbl in enumerate(class_lbls):
        c_inds = np.where(y == c_lbl)[0]

        class_mean = np.mean(x[c_inds, :], 0)

        b_vec = class_mean - mu
        b_vec = np.expand_dims(b_vec, 1)

        s_b = s_b + np.matmul(b_vec, b_vec.T)

        if not use_cov:
            delta = x[c_inds, :] - class_mean
            s_w = s_w + np.matmul(delta.T, delta)
        else:
            s_w = s_w + np.cov(x[c_inds, :], rowvar=False)

    # Calculate weights

    [eig_vls, eig_vecs] = scipy.linalg.eig(s_b, s_w)

    sort_order = np.argsort(np.real(eig_vls))
    keep_inds = sort_order[d_x - n_classes + 1:d_x:1]

    return eig_vecs[:, keep_inds]

