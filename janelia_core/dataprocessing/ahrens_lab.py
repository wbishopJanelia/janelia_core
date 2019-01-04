""" Data tools used for working with results from the Ahrens lab.

    While the tools here were originally developed for working with Ahrens lab data, they
    are written to be generally useful for many datasets.

    William Bishop
    bishopw@hhmi.org

"""

import numpy as np


def extract_swims_from_ephys(ch_data: np.ndarray):
    """ Extracts swim times from ephys data collected from nerve recordings.

    This function is based on code sent by Yu Mu to Will Bishop.

    Args:
        ch_data: A numpy array of shape n_chs by n_smps of electophysiological data.
    """

    # Smooth the data by convolution with a Gaussian kernel
    print('asdfa')

