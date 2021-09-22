""" Contains various useful functions when working with statistics.  """

import numpy as np
import scipy.stats


def get_2d_confidence_ellipse(mn: np.ndarray, cov: np.ndarray, conf: float, n_side_pts: int = 100) -> np.ndarray:
    """ Generates points for plotting a 2-d ellipse containing conf% of probability mass for a Normal distribution.

    Args:
          mn: The mean for the distribution.

          cov: The covariance matrix for the distribution.

          conf: The confidence to produce the ellipse for

          n_side_pts: The number of points to generate for each half of the ellipse.  Must be even.

    Returns:
        pts: The points defining the ellipse.  Each row is a point.  There will 2*n_side_pts total points returned.

    Raises:
        ValueError: If inputs are not the right shape

    """

    if (cov.shape[0] != 2) or (cov.shape[1]) != 2:
        raise(ValueError('cov must be a 2 by 2 matrix.'))
    if len(mn) != 2:
        raise(ValueError('mn must be a vector of length 2.'))

    # Get eigenvalues and eigenvectors of cov
    eig_vls, eig_vecs = np.linalg.eig(cov)

    # Produce ellipse points in axis-aligned coordinates
    s = scipy.stats.chi2.ppf(conf, 2)

    x_max = eig_vls[0]*np.sqrt(s)
    x_pts = np.linspace(-x_max, x_max, n_side_pts)

    y_pts = eig_vls[1]*np.sqrt(np.abs(s - (x_pts/eig_vls[0])**2))

    pts_0 = np.stack([x_pts, y_pts], axis=-1)
    pts_1 = np.flipud(np.stack([x_pts, -1*y_pts], axis=-1))
    pts = np.concatenate([pts_0, pts_1], axis=0)

    # Rotate covariance points back to native coordinate system
    return np.matmul(pts, eig_vecs.transpose()) + mn