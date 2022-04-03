""" Tools for visualizing images.
"""

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg

from janelia_core.visualization.image_generation import generate_2d_fcn_image




def visualize_2d_function(f: Callable, dim_0_range: Sequence[float] = None, dim_1_range: Sequence[float] = None,
                          n_pts_per_dim: Sequence[int] = None, ax = None):
    """ Visualizes a 2-d function.

    Args:
        f: The function to visualize. Should accept input of shape [n by 2], where n is an arbitrary number
        of values and output values of length n.

        dim_0_range: The range of dim 0 values to visualize the function over.  If None, [0.0, 1.0] will be used.

        dim_1_range: The range of dim 1 values to visualize the function over.  If None, [0.0, 1.0] will be used.

        n_pts_per_dim: The number of points to sample per dimension.
        """

    im, dim_0_pts, dim_1_pts = generate_2d_fcn_image(f=f, dim_0_range=dim_0_range, dim_1_range=dim_1_range,
                                                     n_pts_per_dim=n_pts_per_dim)

    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    extent = [dim_0_pts[0], dim_0_pts[-1], dim_1_pts[0], dim_1_pts[-1]]
    im_h = ax.imshow(im.transpose(), extent=extent, origin='lower')
    plt.colorbar(im_h)
    plt.xlabel('Dim 0')
    plt.ylabel('Dim 1')

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




