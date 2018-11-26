""" Tools for viewing predefined ROIs.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QSlider

import pyqtgraph as pg
from pyqtgraph import InfiniteLine
from pyqtgraph import LinearRegionItem

from janelia_core.math.basic_functions import u_th

roi_clrs = np.asarray([[0, 153, 255], # Blue
                      [51, 204, 51], # Green
                      [255, 255, 0], # Yellow
                      [204, 0, 244], # Purple
                      [255, 0, 9]],  # Red
                        dtype=np.uint8)


class TimeLine(QWidget):
    """ QT Widget for viewing time series data.

    The main point of this widget is to allow a user to move select a point in time in data
    and emit signals as the selected time point is changed.
    """

    # This signal emitted as the lines indicating selected time are dragged
    time_dragged = pyqtSignal(float)

    def __init__(self, ts: np.ndarray, vls: np.ndarray):
        """ Creates a new TimeLine object.

        Args:
            ts: A numpy array of time stamps.

            vls: A numpy array of values.  Each row is a different signals.  Columns correspond to
            times in ts.
        """
        super().__init__()

        self.ts = ts
        self.vls = vls

    def init_ui(self):
        """ Initializes the user interface.
        """

        t_min = np.min(self.ts)
        t_max = np.max(self.ts)

        # Create plot widgets
        overview_plot = pg.PlotWidget()
        detail_plot = pg.PlotWidget()

        n_sigs = self.vls.shape[0]
        for s_i in range(n_sigs):
            overview_plot.plot(self.ts, self.vls[s_i, :])
            detail_plot.plot(self.ts, self.vls[s_i, :])

        # Add linear region to overview plot that selects the region shown in the detail plot
        sel_region = LinearRegionItem(movable=True, bounds=[t_min, t_max])
        overview_plot.addItem(sel_region)

        # Make sure as we move either plot, the other is updated
        def update_detailed_region():
            detail_plot.setXRange(*sel_region.getRegion(), padding=0)
        sel_region.sigRegionChanged.connect(update_detailed_region)

        def update_sel_region():
            sel_region.setRegion(detail_plot.getViewBox().viewRange()[0])
        detail_plot.sigXRangeChanged.connect(update_sel_region)

        # Add infinite lines that will indicate where we are in time to both plots
        overview_line = InfiniteLine(movable=True, bounds=[t_min, t_max])
        overview_plot.addItem(overview_line)
        detail_line = InfiniteLine(movable=True, bounds=[t_min, t_max])
        detail_plot.addItem(detail_line)

        # Set things up so our lines are connected when we move them and we
        # emit a signal indicating the lines have been moved
        lines = [overview_line, detail_line]

        def update_other_line_vl(event_line):
            cur_time = event_line.value()
            for ln in lines:
                if ln != event_line:
                    ln.setValue(cur_time)
            self.time_dragged.emit(cur_time)

        overview_line.sigDragged.connect(update_other_line_vl)
        detail_line.sigDragged.connect(update_other_line_vl)

        # Layout everything
        layout = QVBoxLayout()
        layout.addWidget(overview_plot)
        layout.addWidget(detail_plot)

        self.setLayout(layout)

        self.show()


class ROIViewer(QWidget):
    """ An object for viewing ROI information across time within a single plane.

    Rois are displayed on top of a background image. The window will have a slider that allows the user to scroll
    through time.  The window also responds to left and right arrow input to scroll through time.
    """
    sigKeyPress = pyqtSignal(object)

    def __init__(self, bg_img, rois, roi_vl_str='vls', clrs=None):
        """ Creates an ROIViewer object.

        Args:
            bg_img: The background image.  Should be a 2-d numpy array (first dimension y, second dimension x).

            rois: A list of rois.  Each entry is a dictionary with the fields:
                x - a 1 dimensional np.ndarray of x coordinates of pixels in the ROI
                y - a 1 dimensional np.ndarray of y coordinates of pixels in the ROI
                w - a 1 dimensional np.ndarray of weights of each pixel in the ROI
                vls - a 1 dimensional np.ndarray of values the ROI takes on across time.  Should be within 0 to 1.

            roi_vl_str - If the values of the ROI across time are stored in a field with a different name than 'vls',
            the user can specify that with this argument.

            clrs - If not note, a np.ndarray of shape n_rois * 3, where each row specifies the color of an roi.  The dtype
            clrs should be int. And all entries should be within 0 and 255.

        """
        super().__init__()

        self.mn_img = bg_img
        self.rois = rois
        self.roi_vl_str = roi_vl_str
        self.slider = QSlider(Qt.Horizontal)

        if clrs is None:
            n_rois = len(rois)
            clrs = np.zeros([n_rois, 3], dtype=np.int)
            for i in range(n_rois):
                clr_ind = i % roi_clrs.shape[0]
                clrs[i, :] = roi_clrs[clr_ind, :]
        self.clrs = clrs

    def init_ui(self):
        """ Initializes and shows the interface for the ROIViewer."""

        n_tm_pts = len(self.rois[0][self.roi_vl_str])

        # Create a viewbox for our images - this will allow us to pan and scale things
        image_vew_box = pg.ViewBox()

        # Add the background image to the image view
        bg_image_item = pg.ImageItem(self.mn_img)
        image_vew_box.addItem(bg_image_item)

        # Add the rois to the image view
        roi_image_items = [None]*len(self.rois)
        for i, roi in enumerate(self.rois):


            clr = self.clrs[i,:]
            roi_image_items[i] = self._gen_roi_image(i, clr)
            image_vew_box.addItem(roi_image_items[i])

        # Create a graphics view widget, adding our viewbox
        graphics_view = pg.GraphicsView()
        graphics_view.setCentralItem(image_vew_box)

        # Define a helper function to set opacity of rois
        def set_roi_opacity(ind):
            for roi_ind, roi in enumerate(self.rois):
                curVl = roi[self.roi_vl_str][ind]
                roi_image_items[roi_ind].setOpacity(curVl)

        def time_pt_changed(ind):
            set_roi_opacity(ind)

        # Setup the slider
        slider = self.slider
        slider.valueChanged.connect(time_pt_changed)
        slider.setTracking(False)
        slider.setRange(0, n_tm_pts-1)

        # Layout everything
        layout = QVBoxLayout()
        layout.addWidget(graphics_view)
        layout.addWidget(slider)
        self.setLayout(layout)

        # Set things up to respond to key presses
        def process_key_press(ev):
            if ev.key() == Qt.Key_Right:
                slider.setValue(slider.value() + 1)
            elif ev.key() == Qt.Key_Left:
                slider.setValue(slider.value() - 1)

        self.sigKeyPress.connect(process_key_press)

        self.show()

    def set_value(self, idx):
        """ Sets the index of the time point the ROI viewer shows.

        Args:
            idx - the index to show.
        """
        self.slider.setValue(idx)

    def _gen_roi_image(self, roi_ind, clr) -> pg.ImageItem:
        """ Generates an ROI image that can be overlayed the base image.

        Args:
            roi_ind - The index of self.rois of the roi to generate the image for.

            clr - The color the roi should be shown in.  Should be a numpy array of type int and length 3.  All values
            should be between 0 and 255.

        Returns:
            roi_image_item - the image item for the roi
        """
        y_min = np.min(self.rois[roi_ind]['y'])
        y_max = np.max(self.rois[roi_ind]['y'])
        x_min = np.min(self.rois[roi_ind]['x'])
        x_max = np.max(self.rois[roi_ind]['x'])

        y_len = y_max - y_min + 1
        x_len = x_max - x_min + 1

        # Set color of all pixels
        base_image = np.zeros([y_len, x_len, 4], dtype=np.int)
        base_image[:, :, 0:3] = clr

        # Set alpha levels of pixels in the roi
        y_pxls = self.rois[roi_ind]['y'] - y_min
        x_pxls = self.rois[roi_ind]['x'] - x_min
        base_image[y_pxls, x_pxls, 3] = np.ndarray.astype(self.rois[roi_ind]['w']*255, np.int)

        roi_image_item = pg.ImageItem(base_image, autoDownsample=True)

        img_rect = pg.QtCore.QRect(y_min, x_min, y_len, x_len)
        roi_image_item.setRect(img_rect)

        roi_image_item.setOpacity(.2)
        return roi_image_item

    def keyPressEvent(self, ev):
        """ Overrides keyPressEvent of base class, so that we signal to this object that a key has been pressed."""
        self.sigKeyPress.emit(ev)

