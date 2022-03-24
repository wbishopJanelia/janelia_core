""" Contains custom torch functions. """

import torch

# Define constants
LOG_2 = torch.log(torch.tensor(2.0))


def knn_do(x: torch.Tensor, ctrs: torch.Tensor, k: int, m: int, n_ctrs_used: int) -> torch.Tensor:
    """ Searches for the k nearest centers to each point in a set of data, applying dropout to centers.

    This function wraps around knn_mc, applying dropout to the centers.  By this, we mean that the function
    will first randomly select n_ctrs_used centers to use, and then it will only look for nearest neighbors
    among these centers.  Setting n_ctrs_used to the total number of centers results in no dropout.

    Args:

        x: The set of data.  We return centers closest to each point in x.  Of shape n_smps*d

        ctrs: The centers to search among.  Of shape n_ctrs*d

        k: The number of nearest neighbors to search for.

        m: The number of centers to process at a time in the call to knn_mc.  Larger values of m will enable faster
        computation on GPU but use more memory.

        n_ctrs_used: The number of centers to randomly select.

    Returns:

        indices: The indices of the nearest centers for each data point. indices[:,i] are the indices for x[i,:]

    """

    if m > n_ctrs_used:
        m = n_ctrs_used

    n_ctrs = ctrs.shape[0]
    if n_ctrs_used > n_ctrs:
        raise(ValueError('n_ctrs_used is greater than number of centers provided.'))

    if n_ctrs_used == n_ctrs:
        # If we are not using dropout, call knn_mc directly
        return knn_mc(x=x, ctrs=ctrs, k=k, m=m)
    else:
        select_inds = torch.randperm(n_ctrs, device=x.device)
        select_inds = select_inds[0:n_ctrs_used]
        return select_inds[knn_mc(x=x, ctrs=ctrs[select_inds, :], k=k, m=m)]


def knn_mc(x: torch.Tensor, ctrs: torch.Tensor, k: int, m: int) -> torch.Tensor:
    """ Memory constrained version of knn function.

    In this implementation we search for the k nearest centers to each point in a set of data.

    We say it is memory constrained because we can limit the number of centers we search among at a time.  By
    looping through centers in blocks, we can limit the amount of memory used.  Specifically, for any given block, we
    use broadcasting to enable fast calculations, which uses O(n_ctrs*n_data_points) amount of memory, so by limiting
    the number of centers we search through at once, we can limit memory usage.  A post-processing step at the end
    of each loop keeps track of the k-nearest centers seen up until that point in time to each data point.

    Args:

        x: The set of data.  We return centers closest to each point in x.  Of shape n_smps*d

        ctrs: The centers to search among.  Of shape n_ctrs*d

        k: The number of nearest neighbors to search for.

        m: The number of centers to process at a time.  Larger values of m will enable faster computation on GPU but use
        more memory.

    Returns:

        indices: The indices of the nearest centers for each data point. indices[:,i] are the indices for x[i,:]

    Raise:

        ValueError: If k is greater than the number of centers.

    """

    n_ctrs, d = ctrs.shape

    if k > n_ctrs:
        raise(ValueError('k must be less than or equal to the number of centers.'))

    # Determine how many blocks we search for nearest centers over
    n_blocks = int(n_ctrs/m)
    if n_ctrs % m != 0:
        n_blocks += 1

    # Find k-nearest neighbors in a block by block manner
    for b_i in range(n_blocks):
        start_ind = b_i*m
        stop_ind = min(start_ind + m, n_ctrs)
        cur_ctrs = ctrs[start_ind:stop_ind, :]
        n_cur_ctrs = cur_ctrs.shape[0]

        # Find nearest neighbors for this block
        diffs = x - torch.reshape(cur_ctrs, [n_cur_ctrs, 1, d])
        sq_distances = torch.sum(diffs**2, dim=2)

        cur_k = min(k, n_cur_ctrs)
        top_k = torch.topk(sq_distances, k=cur_k, dim=0, largest=False)

        # Compare nearest neighbors for this block to those we find before
        if b_i == 0:
            indices = top_k.indices
            values = top_k.values
        else:
            two_step_indices = torch.cat([indices, top_k.indices + start_ind], dim=0)
            two_step_values = torch.cat([values, top_k.values], dim=0)

            cur_two_step_k = min(k, two_step_indices.shape[0])
            two_step_top_k = torch.topk(two_step_values, k=cur_two_step_k, dim=0, largest=False)

            indices = torch.gather(two_step_indices, 0, two_step_top_k.indices)
            values = torch.gather(two_step_values, 0, two_step_top_k.indices)

    return indices


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """ Computes log cosh(x) in a numerically stable way.

    Args:
        x: Input values

    Returns:
        y: Output values
    """

    abs_x = torch.abs(x)
    return torch.abs(x) - LOG_2 + torch.log1p(torch.exp(-2*abs_x))

