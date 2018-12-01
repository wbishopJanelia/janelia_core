""" Math functions for taking advantage of GPUs.

The functions in this module use pytorch.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
import torch


def truncated_svd(data: np.ndarray, r: int = None, device='cuda') -> list:
    """ Uses GPUs to perform a truncated SVD.

    Args:
        data: the data matrix to perform the svd on

        r: The number of dimensions to return.  If None, r is set to the minimum dimension of the matrix.

    Returns:
        u, d, v: np.ndarrays so that data ~ u*diag(d)*v.T
    """

    if data.shape[0] > data.shape[1]:
        tall_data = True
    else:
        data = data.T
        tall_data = False

    # Compute covariance matrix - we do this on cpu to reduce memory burden on GPU
    print('Computing covariance matrix.')
    cov_m = np.matmul(data.T, data)

    # Move covariance matrix to device
    print('Moving covariance to GPU.')
    cov_m = np.ndarray.astype(cov_m, np.float32)
    cov_m = torch.from_numpy(cov_m)
    cov_m = cov_m.to(device)

    print('Computing eigendecomposition.')
    # Perform eigendecomposition on the covariance matrix
    eig_vecs, eig_vls, _ = torch.svd(cov_m)
    # Get rid of the covariance matrix to free up memory on GPU
    del cov_m

    # Get the largest singular values
    s = torch.sqrt(eig_vls[0:r])

    # Get the corresponding largest eigenvectors, corresponding to left singular vectors
    v = eig_vecs[:, 0:r]

    # Get rid of full eigenvectors and eigenvalues to free up GPU memory
    del eig_vls
    del eig_vecs

    # Now get right singular vectors
    print('Moving data matrix to GPU.')
    data = np.ndarray.astype(data, np.float32) # Convert data to float32 to speed computation
    data = torch.from_numpy(data)
    data = data.to(device)

    print('Computing right singular vectors.')
    m = torch.matmul(v, torch.diag(torch.div(torch.ones(1, device=device), s)))
    u = torch.matmul(data, m)

    print('Moving results to cpu.')
    u = u.cpu().numpy()
    s = s.cpu().numpy()
    v = v.cpu().numpy()

    if tall_data:
        return [u, s, v]
    else:
        return [v, s, u]
    #return s
