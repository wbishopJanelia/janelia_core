""" Tools for viewing predefined ROIs.

    William Bishop
    bishopw@hhmi.org
"""

import warnings

import numpy as np
import numpy.matlib
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QSlider

import pyqtgraph as pg
from pyqtgraph import InfiniteLine
from pyqtgraph import LinearRegionItem
from pyqtgraph import makeARGB

from janelia_core.math.basic_functions import l_th
from janelia_core.math.basic_functions import u_th
from janelia_core.visualization.utils import alpha_overlay

roi_clrs = np.asarray([[0, 153, 255], # Blue
                      [255, 0, 9], # Red
                      [51, 204, 51], # Green
                      [255, 255, 0], # Yellow
                      [204, 0, 244]], # Purple
                        dtype=np.uint8)


class StaticROIViewer(QWidget):
    """ QT Widget for viewing static 3D roi data.

    This widget allows the user to provide a 3D background image as well as a set of 3D ROIs. The user
    can then view slices through the 3D volume.
    """

    sigKeyPress = pyqtSignal(object)

    def __init__(self, bg_image: np.ndarray, rois: list, dim: int = 0, clrs: np.ndarray = None,
                 weight_gain: float = 1.0, levels: list = None, ignore_warnings: bool = False):
        """ Creates a new StaticROIViewer object.

        Args:
            bg_image: The background image to view.

            rois: A list of ROI objects.

            dim: The dimension to slice along.

            clrs: A n_roi*3 array.  Each row gives the color to plot the corresponding ROI in.  If this is none,
            random colors will be assigned to ROIs.

            weight_gain: The value to multiply weights of rois by before mapping the weights to aplha values in the
            range of 0 to 255.

            levels: If none, min, max intensity levels for each plane are automatically set.  Otherwise,
            levels should be a sequence giving the [min, max] levels

            ignore_warning: True if warnings should not be generated
        """

        super().__init__()

        self.bg_image = bg_image
        self.rois = rois
        self.dim = dim
        self.weight_gain = weight_gain

        self.plane_images = None
        self.slider = QSlider(Qt.Horizontal)
        self.levels = levels
        self.ignore_warnings = ignore_warnings

        if clrs is None:
            n_rois = len(rois)
            clrs = np.zeros([n_rois, 3], dtype=np.int)
            for i in range(n_rois):
                clr_ind = i % roi_clrs.shape[0]
                clrs[i, :] = roi_clrs[clr_ind, :]
        self.clrs = clrs

    # Set things up to respond to key presses
    def process_key_press(self, ev):
        if ev.key() == Qt.Key_Right:
            self.slider.setValue(self.slider.value() + 1)
        elif ev.key() == Qt.Key_Left:
            self.slider.setValue(self.slider.value() - 1)

    def init_ui(self):
        """ Initializes the user interface.
        """

        # First thing we need to do is create the set of images to display in each slice - we do blending of
        # rois on top of the background image for each plane here

        image_shape = self.bg_image.shape
        n_planes = image_shape[self.dim]

        slices = [[slice(0, image_shape[d], 1) if d != self.dim else slice(p, p+1, 1) for d in range(3)]
                  for p in range(n_planes)]
        slices = [tuple(slices[p]) for p in range(n_planes)]

        self.plane_images = list()
        for p in range(n_planes):
            plane_image = np.squeeze(self.bg_image[slices[p]])
            plane_image = np.stack([plane_image, plane_image, plane_image], 2)

            for roi_idx, roi in enumerate(self.rois):
                sliced_roi = roi.slice_roi(p, self.dim, retain_dim=False)

                n_roi_voxels = len(sliced_roi.weights)
                roi_clr = self.clrs[roi_idx, :]
                roi_clr = np.matlib.repmat(roi_clr, n_roi_voxels, 1)

                w = self.weight_gain*sliced_roi.weights
                if len(w) > 0:
                    if np.min(w) < 0:
                        if not self.ignore_warnings:
                            warnings.warn('Some weights for roi ' + str(roi_idx) + ' are less than 0.')
                        w = l_th(w, 0)
                    if np.max(w) > 1:
                        if not self.ignore_warnings:
                            warnings.warn('Some weights for roi ' + str(roi_idx) + ' are greater than 1.')
                        w = u_th(w,0)

                plane_image = alpha_overlay(plane_image, sliced_roi.voxel_inds, roi_clr,
                                            255*self.weight_gain*sliced_roi.weights)

            self.plane_images.append(plane_image)

        # Create a viewbox for our images - this will allow us to pan and scale things
        image_aspect_ratio = self.plane_images[0].shape[0] / self.plane_images[0].shape[1]
        image_vew_box = pg.ViewBox(lockAspect=image_aspect_ratio)

        # Add the plane 0 image to the image view
        image_item = pg.ImageItem(self.plane_images[0])
        if self.levels is not None:
            image_item.setLevels(self.levels, update=False)
        image_vew_box.addItem(image_item)

        # Create a graphics view widget, adding our viewbox
        graphics_view = pg.GraphicsView()
        graphics_view.setCentralItem(image_vew_box)

        def slider_moved(vl):
            image_item.setImage(self.plane_images[vl], autoLevels=False)

        # Setup things to hand key presses
        self.sigKeyPress.connect(self.process_key_press)

        # Setup the slider
        slider = self.slider
        slider.valueChanged.connect(slider_moved)
        slider.setTracking(True)
        slider.setRange(0, n_planes - 1)

        # Layout everything
        layout = QVBoxLayout()
        layout.addWidget(graphics_view)
        layout.addWidget(slider)
        self.setLayout(layout)

        self.show()

    def keyPressEvent(self, ev):
        """ Overrides keyPressEvent of base class, so that we signal to this object that a key has been pressed."""
        self.sigKeyPress.emit(ev)

    def set_plane(self, p):
        """ Sets the plane of the viewer shows.

        Args:
            idx - the index to show.
        """
        self.slider.setValue(p)


class TimeLine(QWidget):
    """ QT Widget for viewing time series data.

    The main point of this widget is to allow a user to move select a point in time in data
    and emit signals as the selected time point is changed.
    """

    # This signal emitted as the lines indicating selected time are dragged
    time_dragged = pyqtSignal(float)

    def __init__(self, ts: np.ndarray, vls: np.ndarray, clrs: np.ndarray = None):
        """ Creates a new TimeLine object.

        Args:
            ts: A numpy array of time stamps.

            vls: A numpy array of values.  Each column is a different signals.  Rows correspond to
            times in ts.

            clrs: A n_signals*3 array, where n_signals is the number of signals in vls.  Each row
            gives the color to plot the corresponding signal in vls in.  If this is None, all
            signals will be white.

        """
        super().__init__()

        self.ts = ts

        if len(vls.shape) == 1:
            vls = np.reshape(vls, [vls.size, 1])

        self.vls = vls

        if clrs is None:
            n_sigs = vls.shape[1]
            self.clrs = 255*np.ones([n_sigs, 3], dtype=int)
        else:
            self.clrs = clrs

    def init_ui(self):
        """ Initializes the user interface.
        """

        t_min = np.min(self.ts)
        t_max = np.max(self.ts)

        # Create plot widgets
        overview_plot = pg.PlotWidget()
        detail_plot = pg.PlotWidget()

        n_sigs = self.vls.shape[1]
        for s_i in range(n_sigs):
            overview_plot.plot(self.ts, self.vls[:, s_i], pen=self.clrs[s_i, :])
            detail_plot.plot(self.ts, self.vls[:, s_i], pen=self.clrs[s_i, :])

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

    def __init__(self, bg_img, rois, roi_vl_str='vls', clrs=None, title:str=None, roi_scale_f: np.float64=None):
        """ Creates an ROIViewer object.

        Args:
            bg_img: The background image.  Should be a 2-d numpy array (first dimension y, second dimension x).

            rois: A list of ROI objects. These should have a 'vls' field which stores the values of the ROI
            across time.  vls should be between 0 and 1.

            roi_vl_str - If the values of the ROI across time are stored in a field with a different name than 'vls',
            the user can specify that with this argument.

            clrs - If not none, a np.ndarray of shape n_rois * 3, where each row specifies the color of an roi.  The dtype
            clrs should be int. And all entries should be within 0 and 255.

            title - An optional title to provide for the window

            roi_scale_f: If provided, the weights of all rois are divided by this factor to determine there alpha value
            (which must be between 0 and 1).  If none, each roi is scaled individually by its max weight.

        """
        super().__init__()

        self.mn_img = bg_img
        self.rois = rois
        self.roi_vl_str = roi_vl_str
        self.slider = QSlider(Qt.Horizontal)
        self.title = title
        self.roi_scale_f = roi_scale_f

        for roi_ind, roi in enumerate(rois):
            vls = getattr(roi, roi_vl_str)
            min_vl = np.min(vls)
            max_vl = np.max(vls)
            if min_vl < 0 or max_vl > 1:
                warnings.warn(RuntimeWarning('Roi ' + str(roi_ind) + ' has values outside of [0, 1].'))

        if clrs is None:
            n_rois = len(rois)
            clrs = np.zeros([n_rois, 3], dtype=np.int)
            for i in range(n_rois):
                clr_ind = i % roi_clrs.shape[0]
                clrs[i, :] = roi_clrs[clr_ind, :]
        self.clrs = clrs

    def init_ui(self):
        """ Initializes and shows the interface for the ROIViewer."""

        if self.title is not None:
            self.setWindowTitle(self.title)

        n_tm_pts = len(getattr(self.rois[0], self.roi_vl_str))

        # Create a viewbox for our images - this will allow us to pan and scale things
        imageAspectRatio = self.mn_img.shape[0]/self.mn_img.shape[1]
        image_vew_box = pg.ViewBox(lockAspect=imageAspectRatio)

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
                cur_vl = getattr(roi, self.roi_vl_str)[ind]
                roi_image_items[roi_ind].setOpacity(cur_vl)

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
        roi = self.rois[roi_ind]

        # Get bounding box for roi
        bounding_box = roi.bounding_box()
        side_lengths = roi.extents()
        dim_starts = [s.start for s in bounding_box]

        # Get coordinates of voxels in ROI within bounding box
        roi_voxel_inds = roi.list_all_voxel_inds()
        n_roi_dims = len(side_lengths)
        shifted_roi_voxel_inds = tuple([roi_voxel_inds[d] - dim_starts[d] for d in range(n_roi_dims)])

        # Get rid of first dimension if we need
        if n_roi_dims > 2:
            side_lengths = side_lengths[1:]
            dim_starts = dim_starts[1:]
            shifted_roi_voxel_inds = shifted_roi_voxel_inds[1:]

        # Set color of all pixels
        base_image = np.zeros([*side_lengths, 4], dtype=np.int)
        base_image[:, :, 0:3] = clr

        # Set alpha levels of pixels in the roi
        if self.roi_scale_f is None:
            norm_w = roi.list_all_weights()/np.max(roi.list_all_weights())
        else:
            norm_w = roi.list_all_weights()/self.roi_scale_f
        base_image[shifted_roi_voxel_inds[0], shifted_roi_voxel_inds[1], 3] = np.ndarray.astype(norm_w*255, np.int)

        roi_image_item = pg.ImageItem(base_image, autoDownsample=True)
        roi_image_item.setLevels([0, 255])

        img_rect = pg.QtCore.QRect(dim_starts[0], dim_starts[1], side_lengths[0], side_lengths[1])
        roi_image_item.setRect(img_rect)

        roi_image_item.setOpacity(.5)
        return roi_image_item

    def keyPressEvent(self, ev):
        """ Overrides keyPressEvent of base class, so that we signal to this object that a key has been pressed."""
        self.sigKeyPress.emit(ev)


class MultiPlaneROIViewer():
    """ An object for viewing ROIs across multiple planes and roi groups.

    This object allows for the visualization of a set of signals along with roi values across time.  The main window
    consists of a TimeLine Widget showing the signals. There will be additional windows to show ROIs in each plane and
    for each group. As the user moves the slider on this widget, roi values for the time the slider
    corresponds to will be updated in the opacity of rois.
    """

    def __init__(self, ts: np.ndarray, vls: np.ndarray, roi_groups: list, bg_imgs: list, planes: list, clrs: list = None,
                 roi_signals:list = None, roi_group_names: list=None, roi_scale_fs: list=None):
        """ Create a MultiPlaneROIViewer object.

        Args:
            ts: A numpy array of time stamps.

            vls: A numpy array of values for the main timeline.  Each row is a different signal.  Columns correspond to
            times in ts.

            roi_groups: A list of roi_groups (e.g., neurons and glia).  Each list contains roi objects for the group.  These
            roi objects should be supplemented with a 'vls' attribute containing the value of the roi at each point in ts.

            bg_imgs: A list of back ground images.  bg_imgs[i] contains the background image to use for rois[i]

            planes: A list. planes[i] is the indices of planes to show for rois in roi_groups[i]

            clrs: A list.  clrs[i] contains a shape n_rois * 3 np.ndaray, where each row specifies the color of the
            corresponding roi in rois[i].  The dtype of clrs should be int, and all entries should be within 0 and 255.
            If clrs is None, then random colors will be assigned to the ROIs.

            roi_signals: A list.  roi_signals[i] contains indices of rois in roi_groups[i] whose values should be plotted
            along with those in the vls array.  If None, no roi values will be plotted.

            roi_group_names: An optional list providing a name for each roi group.

            roi_scale_fs: If provided, a list.  Each entry contains a factor by which the weights of all rois in
            the corresponding group are divided by to determine there alpha value (which must be between 0 and 1).
            If none, each roi is scaled individually by its max weight.

            """

        self.ts = ts
        self.roi_groups = roi_groups
        self.bg_imgs = bg_imgs
        self.planes = planes
        self.roi_signals = roi_signals
        self.roi_group_names = roi_group_names

        if roi_scale_fs is None:
            roi_scale_fs = [None]*len(roi_group_names)
        self.roi_scale_fs = roi_scale_fs

        if len(vls.shape) == 1:
            vls = np.reshape(vls, [vls.size, 1])
        self.vls = vls

        if clrs is not None:
            self.clrs = clrs
        else:
            clrs = list()
            cnt = 0
            for grp in roi_groups:
                n_rois = len(grp)
                grp_clrs = np.zeros([n_rois, 3], dtype=np.int)
                for i in range(n_rois):
                    clr_ind = cnt % roi_clrs.shape[0]
                    grp_clrs[i, :] = roi_clrs[clr_ind, :]
                    cnt += 1
                clrs.append(grp_clrs)
            self.clrs = clrs

    def init_ui(self):
        """ Initializes the UI. """

        # Add roi signals to those we show in timeline if user has specified this
        if self.roi_signals is not None:
            roi_signals = list()
            roi_clrs = list()
            if self.roi_signals is not None:
                for grp_ind, grp in enumerate(self.roi_groups):
                    for roi_ind in self.roi_signals[grp_ind]:
                        roi_signals.append(grp[roi_ind].vls)
                        roi_clrs.append(self.clrs[grp_ind][roi_ind,:])
            roi_signals = np.asarray(roi_signals).T
            roi_clrs = np.asarray(roi_clrs)

            sig_values = np.concatenate([self.vls, roi_signals], 1)
            sig_clrs = np.concatenate([255*np.ones([self.vls.shape[1], 3]), roi_clrs], 0)
        else:
            sig_values = self.vls
            sig_clrs = 255*np.ones([self.vls.shape[1], 3])

        # Create timeline
        tl = TimeLine(self.ts, sig_values, sig_clrs)
        tl.init_ui()

        # Create ROI Viewers for each plane in each roi group
        def gen_roi_viewer(bg_image, grp, plane, grp_clrs, grp_name, roi_scale_f):
            """ Helper function to create roi viewers for each plane and roi group and to
            connect them to the timeline."""
            plane_rois = list()
            plane_clrs = list()
            for roi_ind, roi in enumerate(grp):
                if roi.intersect_plane(plane):
                    plane_roi = roi.slice_roi(plane)
                    plane_roi.vls = roi.vls
                    plane_rois.append(plane_roi)
                    plane_clrs.append(grp_clrs[roi_ind,:])
            plane_clrs = np.asarray(plane_clrs)

            rv = ROIViewer(bg_image[plane, :, :], plane_rois, clrs=plane_clrs,
                           title=grp_name + ': ' + str(plane), roi_scale_f=roi_scale_f)
            rv.init_ui()

            def update_rv_time(ev):
                small_inds = np.nonzero(self.ts - ev < 0)[0]
                if small_inds.size == 0:
                    idx = 0
                else:
                    idx = small_inds[-1]
                rv.set_value(idx)

            tl.time_dragged.connect(update_rv_time)

        for grp_ind, grp in enumerate(self.roi_groups):
            for plane in self.planes[grp_ind]:
                if self.roi_group_names is not None:
                    group_name = self.roi_group_names[grp_ind]
                else:
                    group_name = 'Plane'
                gen_roi_viewer(self.bg_imgs[grp_ind], grp, plane, self.clrs[grp_ind], group_name, self.roi_scale_fs[grp_ind])