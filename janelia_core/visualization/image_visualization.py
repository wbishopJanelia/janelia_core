""" Tools for visualizing images.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg


def signed_max_project(volume: np.ndarray, axis: int):
    """ Performs a signed max projection on 3-d data.

    Args:
        volume: The volume to do the max projection on

        axis: The axis to do the max projection along.

    Returns:
        im: The max projection image.  A 2-d array with dimensions inherited from volume.

    Raises:
        ValueError: If volume is not a 3d array
    """

    if volume.ndim != 3:
        raise(ValueError('volume must be a 3-d array.'))

    inds = np.argmax(a=np.abs(volume), axis=axis)

    ret_shape = list(volume.shape)
    ret_shape[axis] = 1
    m_grid = np.meshgrid(*[np.arange(v) for v in ret_shape], indexing='ij')
    m_grid = [m.squeeze() for m in m_grid]
    m_grid[axis] = inds
    m_grid = tuple(m_grid)

    return volume[m_grid]


class GroupedStackedImageVisualizer(QWidget):
    """ A QWidget for viewing a set of stacked images.

    This object creates a set of ImageViews, side by side.  Each image stack can
    be seen in it's own image view, but as you move through the image stack in one
    image view, the other image views are also updated.

    """

    def __init__(self, imgs: list, titles: list = None, color_maps = None):
        """ Creates a GroupedStackedImageVisualizer object.

        Args:
            imgs: A list of images.  Each entry contains an image to display.

            titles: A list of titles for the images. If None, no special titles will be display.

            color_maps: Either (1) a list of colormaps or (2) a single colormp or (3) None.
            If a list of colormaps, the list gives the colormaps for each image.  If a single
            colormap, the colormap that will be used for all images.  If None, default
            colormaps will be used.

        """

        super().__init__()

        self.imgs = imgs
        self.titles = titles
        self.color_maps = color_maps

    def init_ui(self, levels_range: Sequence = None):
        """ Initializes the UI for a GroupedStackedImageVisualizer object.

        Args:
            levels_range: range[0] gives the lower level and range[1] gives the upper
            range to set levels to.  If none, levels will be set automatically.
        """

        n_imgs = len(self.imgs)

        custom_color_maps = False
        if self.color_maps is not None:
            if isinstance(self.color_maps, pg.ColorMap):
                color_maps = [self.color_maps]*n_imgs
            else:
                color_maps = self.color_maps
            custom_color_maps = True

        custom_titles = self.titles is not None

        # Create the main window the image views will go in
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        # Add images
        im_views = [None]*n_imgs
        for img_i, img in enumerate(self.imgs):

            if custom_titles:
                im_title = QtGui.QLabel(self.titles[img_i])
                layout.addWidget(im_title, 0, img_i)

            im_view = pg.ImageView()
            im_view.setImage(img)

            # Set initial levels automatically
            hist_widget = im_view.getHistogramWidget()
            if levels_range is None or levels_range[img_i] is None:
                low_p = np.percentile(img, 2)
                high_p = np.percentile(img, 98)
                hist_widget.setLevels(low_p, high_p)
            else:
                hist_widget.setLevels(*levels_range[img_i])

            im_views[img_i] = im_view
            if custom_color_maps:
                if color_maps[img_i] is not None:
                    im_view.setColorMap(color_maps[img_i])
            layout.addWidget(im_view, 1, img_i)

        # Wire everything up to move together when a slider is moved
        for i in range(n_imgs):
            for j in range(n_imgs):
                im_views[i].sigTimeChanged.connect(im_views[j].setCurrentIndex)

        self.show()


