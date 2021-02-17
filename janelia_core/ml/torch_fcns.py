""" Contains functions for use with pytorch. """

import torch


def knn_mc(x: torch.Tensor, ctrs: torch.Tensor, k: int, m:int) -> torch.Tensor:
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

        m: The number of centers to process at a time.  This should be in the range 1 <= m <= n_ctrs. Larger values of m
        will enable faster computation on GPU but use more memory.

    Returns:

        indices: The indices of the nearest centers for each data point. indices[:,i] are the indices for x[i,:]

    """

    n_ctrs, d = ctrs.shape

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

