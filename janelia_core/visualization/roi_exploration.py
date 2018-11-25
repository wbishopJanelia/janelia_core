""" Tools for viewing predefined ROIs.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QSlider

import pyqtgraph as pg
from pyqtgraph import InfiniteLine
from pyqtgraph import LinearRegionItem


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

    def __init__(self, mn_img, rois):
        super().__init__()

        self.mn_img = mn_img
        self.rois = rois

    def init_ui(self):

        n_tm_pts = len(self.rois[0]['vls'])


        # Create a viewbox for our images - this will allow us to pan and scale things
        image_vew_box = pg.ViewBox()

        # Add the background image to the image view
        bg_image_item = pg.ImageItem(self.mn_img)
        image_vew_box.addItem(bg_image_item)

        # Add the rois to the image view
        roi_image_items = [None]*len(self.rois)
        for i, roi in enumerate(self.rois):

            clr_ind = i%roi_clrs.shape[0]
            clr = roi_clrs[clr_ind,:]

            roi_image = self._gen_roi_image(i, clr)
            roi_image_items[i] = pg.ImageItem(roi_image)
            roi_image_items[i].setOpacity(.2)
            image_vew_box.addItem(roi_image_items[i])

        # Create a graphics view widget, adding our viewbox
        graphics_view = pg.GraphicsView()
        graphics_view.setCentralItem(image_vew_box)

        # Define a helper function to set opacity of rois
        def set_roi_opacity(ind):
            for roi_ind, roi in enumerate(self.rois):
                curVl = roi['vls'][ind]
                roi_image_items[roi_ind].setOpacity(curVl)

        # Create a slider
        def time_pt_changed(ind):
            print("Slider moved:" + str(ind))
            set_roi_opacity(ind)

        slider = QSlider(Qt.Horizontal)
        slider.valueChanged.connect(time_pt_changed)
        slider.setTracking(False)
        slider.setRange(0, n_tm_pts-1)


        # Layout everything
        layout = QVBoxLayout()
        layout.addWidget(graphics_view)
        layout.addWidget(slider)
        self.setLayout(layout)

        self.show()

    def _gen_roi_image(self, roi_ind, clr):

        # Set color of all pixels
        base_image_shape = self.mn_img.shape
        base_image = np.zeros([*base_image_shape, 4], dtype=np.int)
        base_image[:, :, 0:3] = clr

        # Set alpha levels of roi pixels (non-roi pixels are zero)
        x_pxls = self.rois[roi_ind]['x']
        y_pxls = self.rois[roi_ind]['y']
        base_image[y_pxls, x_pxls, 3] = np.ndarray.astype(self.rois[roi_ind]['w']*255, np.int)
        return base_image
