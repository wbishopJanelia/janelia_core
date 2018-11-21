""" Tools for viewing predefined ROIs.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


def visualize_rois(mean_img: np.ndarray, rois: list, roi_vls: np.ndarray):
    """ Generates a GUI for viewing ROI data through time.

    Args:
        mean_img: A mean image to plot under the ROIs.

        rois: A list.  Each entry contains a dictionary with the entries 'x' and 'y' which contain
        the x and y coordinates of pixels in the ROI.

        roi_vls: A numpy array of shape n_roi by t.  roi_vls[r,:] contain the values for rois[r]
        through time.
    """
    app = pg.mkQApp()
    w = QtGui.QWidget()

