""" Data tools used for working with results from the Ahrens lab.

    While the tools here were originally developed for working with Ahrens lab data, they
    are written to be generally useful for many datasets.

    William Bishop
    bishopw@hhmi.org

"""

from typing import Tuple
import warnings

import numpy as np
import scipy.stats as stats


from janelia_core.math.basic_functions import find_binary_runs

def down_sample_ephys_vls(ts: np.ndarray, image_acq_starts: np.ndarray, n_images: int, vls: np.ndarray,
                          ds_type: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    A function to take raw ephys data and down-sample it to the imaging rate of an experiment.

    Values are down-sampled between image acquisitions, so that the down-sampled value for image i will be a function
    of the data that falls between the start of acquisition of image i and the start of acquisition for image i+1.

    Args:
        ts: Time stamps for the raw ephys data.

        image_acq_starts: A binary array with a value of 1 indicating each point in time acquistion image starts

        n_images: The number of images that were acquired.

        vls: The raw ephys values to down sample

        ds_type: The type of down-sampling to perform. Options are:

            'mean' The down-sampled value is the mean value between the start of image acquisitions.

            'mode' The down-sampled value is the mode between the start of image acquisitions.

            'consant' The down-sampled value is the constant value between the start of image acquisitions.  If
            selected, a check will be performed to ensure values are actually constant, and if this check is failed,
            an error will be raised.

    Returns:

        ds_ts: The time stamps of the down sampled values.  Time stamps correspond to the start of image acquisitions.

        ds_vls: The down sampled values.

    Raises:
        ValueError: If n_images is greater than the number of image starts in image_acq_starts.  We require n_images to
        be strictly less than the number of image acquisition starts so a period of image acquisition is well defined
        for the last requested image.

        ValueError: If 'constant' down-sampling is selected, but non-constant values are detected in a sampling period.

    """
    n_image_starts = np.sum(image_acq_starts == 1)

    # Get slices of ephys values we consider for each slice
    image_inds = np.nonzero(image_acq_starts)[0]
    start_inds = image_inds[0:n_images]

    if n_images < n_image_starts:
        stop_inds = image_inds[1:n_images+1]
    elif n_images == n_image_starts:
        stop_inds = start_inds[1:]
        image_inds_diff = stats.mode(np.diff(image_inds)).mode[0]
        stop_inds = np.append(stop_inds, stop_inds[-1] + image_inds_diff)
        warnings.warn('n_images == number of images start, estimating end time of last image acquisition.')
    else:
        raise(ValueError('n_images is greater than number of image starts found in image_acq_starts'))

    ind_slices = [slice(start, stop) for start, stop in zip(start_inds, stop_inds)]

    # Perform down-sampling here
    ds_ts = np.zeros(n_images)
    ds_vls = np.zeros(n_images)
    for i, sl in enumerate(ind_slices):
        ds_ts[i] = ts[sl.start]
        if ds_type == 'mean':
            ds_vls[i] = np.mean(vls[sl])
        elif ds_type == 'constant':
            c_vl = (vls[sl])[0]
            if np.any(vls[sl] != c_vl):
                raise(ValueError('Caught non-constant value in slice ' + str(sl)))
            else:
                ds_vls[i] = c_vl
        elif ds_type == 'mode':
            ds_vls[i] = stats.mode(vls[sl]).mode[0]
        else:
            raise(ValueError('The ds_type ' + ds_type + ' is not recogonized.'))

    return ds_ts, ds_vls


def find_camera_triggers_in_ephys(sig: np.ndarray, th: float = 3.8, smp_tol: int = 2) -> np.ndarray:
    """ Finds indices where the camera was triggered in the raw ephys data.

    Args:
        sig: 1-d array of camera trigger data

        th: the threshold that indicates a camera threshold

        smp_tol: After extracting camera triggers, we do a check and make sure all camera triggers are the
        same number of samples apart +/- a tolerance; this is the tolerance

    Returns:
        inds: The indices of sig where values first pass th

    Raises:
        ValueError: If the first value of sig is greather than th (since then timing of the onset cannot be
        determined)

        RuntimeError: If the tolerance check is not passed

    """

    if sig[0] > th:
        raise(ValueError('First value of signal greater than signal.'))

    runs = find_binary_runs(sig > th)
    inds = np.asarray([s.start for s in runs])

    # Check to make sure we pass a data integrity check
    inds_diff = np.diff(inds)
    diff_mode = stats.mode(inds_diff).mode[0]
    if np.any(np.abs(inds_diff - diff_mode) > smp_tol):
        raise(RuntimeError('Extracted camera triggers failed timig tolerance check.'))

    return inds