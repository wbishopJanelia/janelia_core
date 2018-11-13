""" Tools for visualizing imaging experiment data.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np
import pyspark
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import types

import janelia_core.dataprocessing.dataset
from janelia_core.dataprocessing.utils import get_processed_image_data


def visualize_exp(dataset: janelia_core.dataprocessing.dataset.DataSet,
                  cont_var_key: str, image_keys: list, cont_var_inds: list = None, cont_var_labels: list = None,
                  func: types.FunctionType = None, sc: pyspark.SparkContext = None) -> QtGui.QWidget:
    """ Function for visually exploring the data in a Dataset object.

    This function will create a GUI that plots continuous variables through time, allowing
    the user to zoom in on the continuous variables.  After zooming in on a region of the
    continuous variables, the user can load the images for that region.

    Args:
        dataset: The dataset to explore.

        cont_var_key: The key of the dictionary containing continuous variables to examine.

        image_keys: A list of keys to dictionaries containing image series to examine.

        cont_var_inds: A list of indices of continuous variables to plot. If None, all variables will be plotted.

        cont_var_labels: A list of labels for each continuous variable. If None, no labels will be generated.

        func: A function to apply when processing each image.  If none, max projection in z will be used.

        sc: A spark context, which can be optionally provided to speed up loading images.

    Returns: The created window.
    """

    ephys_time_stamps = dataset.ts_data[cont_var_key]['ts']
    ephys_data = dataset.ts_data[cont_var_key]['vls']

    if cont_var_inds is None:
        cont_var_inds = [i for i in range(ephys_data.shape[1])]

    app = pg.mkQApp()
    w = QtGui.QWidget()

    # Prepare the plots
    p1 = pg.PlotWidget()
    if cont_var_labels is not None:
        leg = p1.addLegend()
    p2 = pg.PlotWidget()
    for i, var_ind in enumerate(cont_var_inds):
        if cont_var_labels is not None:
            p1.plot(ephys_time_stamps, ephys_data[:, var_ind], name=cont_var_labels[i])
        else:
            p1.plot(ephys_time_stamps, ephys_data[:, var_ind])
        p2.plot(ephys_time_stamps, ephys_data[:, var_ind])

    # Setup things to allow zooming
    lr = pg.LinearRegionItem([0, 1])
    lr.setZValue(-10)
    p1.addItem(lr)

    def update_plot():
        p2.setXRange(*lr.getRegion(), padding=0)

    def update_region():
        lr.setRegion(p2.getViewBox().viewRange()[0])

    lr.sigRegionChanged.connect(update_plot)
    p2.sigXRangeChanged.connect(update_region)

    windows = []

    def load_images():
        selected_region = lr.getRegion()
        selected_dataset = dataset.select_time_range(selected_region[0], selected_region[1])
        selected_cont_series = selected_dataset.ts_data[cont_var_key]
        selected_image_series = [selected_dataset.ts_data[k] for k in image_keys]
        windows.append(view_images_with_continuous_values(selected_cont_series, selected_image_series, func,
                                           cont_var_inds, cont_var_labels, image_keys, sc))

    # Add button so we can load selected data
    btn = pg.QtGui.QPushButton("Load Images.")
    btn.clicked.connect(load_images)

    # Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    layout.addWidget(p1, 0, 0)
    layout.addWidget(p2, 1, 0)
    layout.addWidget(btn, 2, 0)

    w.show()
    QtGui.QApplication.instance().exec_()
    return w


def view_images_with_continuous_values(cont_var_dict: dict, image_dicts: list, func: types.FunctionType = None,
                                       cont_var_inds: list = None, cont_var_labels: list = None,
                                       image_labels: list = None, sc: pyspark.SparkContext = None) -> QtGui.QWidget:
    """ Function to explore continuous variables and series of images together.

    There will be a slider on the bottom of the GUI for the continuous variable data which the user can move to scroll
    through time.  This slider will scroll through time for all image series.  Additionally, there are sliders for each
    individual time series.

    Note: Continuous variables and image series must have the same time stamps.
    
    Args: 
        cont_var_dict: A dictionary containing data for a time series of continuous values.  Must have two keys.  
        The first, 'ts',is a 1-d numpy.ndarray with timestamps.  The second 'vls' is a list of or numpy array of data
        for each point in ts, where each row of data is a vector of continuous values for a time point.
        
        image_dicts: A list of dictionaries. Each dictionary contains data for one time series of images. These
        dictionaries must have a 'ts' key, with timestamps (as cont_var_dict) and a 'vls' key, which will contain
        either (1) a 3-d numpy.ndarray of dimensions n_ts*n_pixels_n_pixels or (2) a list of paths to image files of
        length n_ts.

        func: A function to apply when processing each image.  If none, max projection in z will be used.

        cont_var_inds: A list of indices of continuous variables to plot.

        cont_var_labels: A list of strings with the labels for each continuous variable (each column in 'vls' of the
        cont_var_dict).

        image_labels: A list of strings with labels for each series of images in image_dicts.

        sc: A spark context, which can be optionally provided to speed up loading images.

        Returns: The created window.
    """

    time_stamps = cont_var_dict['ts']
    n_time_stamps = len(time_stamps)
    time_min = np.min(time_stamps)
    time_max = np.max(time_stamps)
    current_index = [0]

    # Read in image data
    if func is None:
        func = lambda img: np.max(img, 0)
    img_data = [np.asarray(get_processed_image_data(img_dict['vls'], func, sc)) for img_dict in image_dicts]

    app = pg.mkQApp()
    w = QtGui.QWidget()

    # Prepare plot of continuous variables
    cont_var_win = KeyPressWindow()
    cont_var_plot = cont_var_win.addPlot()
    cont_var_plot.setLimits(xMin=time_min - 1, xMax=time_max + 1)
    if cont_var_labels is not None:
        cont_var_plot.addLegend()
    for i, var_ind in enumerate(cont_var_inds):
        if cont_var_labels is not None:
            cont_var_plot.plot(cont_var_dict['ts'], cont_var_dict['vls'][:, var_ind], name=cont_var_labels[i])
        else:
            cont_var_plot.plot(cont_var_dict['ts'], cont_var_dict['vls'][:, var_ind])

    cont_line = pg.InfiniteLine(movable=True)
    cont_line.setBounds([time_min, time_max])
    cont_var_plot.addItem(cont_line)

    # Prepare the image plots
    im_views = []
    for i, data in enumerate(img_data):
        imv = pg.ImageView()
        imv.setImage(data)
        im_views.append(imv)
        if image_labels is not None:
            imv.view.addItem(pg.TextItem(image_labels[i]))

    # Set things up so when position of the continuous time line is changed,
    # the image plots are also updated
    def process_cont_line_change():
        smaller_time_stamps = np.flatnonzero(time_stamps - cont_line.value() < 0)
        if len(smaller_time_stamps) == 0:
            img_ind = 0
        else:
            img_ind = smaller_time_stamps[-1]

        current_index[0] = img_ind
        for imv in im_views:
            imv.setCurrentIndex(img_ind)

    cont_line.sigPositionChangeFinished.connect(process_cont_line_change)

    def process_cont_var_keypress(event):
        if event.key() == QtCore.Qt.Key_Right:
            current_index[0] = min(current_index[0] + 1, n_time_stamps - 1)
        elif event.key() == QtCore.Qt.Key_Left:
            current_index[0] = max(current_index[0] - 1, 0)
        cont_line.setValue(time_stamps[current_index[0]])
        for imv in im_views:
            imv.setCurrentIndex(current_index[0])

    cont_var_win.sigKeyPress.connect(process_cont_var_keypress)

    # Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)
    for i, iv in enumerate(im_views):
        layout.addWidget(iv, 0, i)

    layout.addWidget(cont_var_win, 1, 0, 1, 2)
    layout.setRowStretch(0, 10)
    layout.setRowStretch(1, 3)

    w.show()

    QtGui.QApplication.instance().exec_()

    return w


class KeyPressWindow(pg.GraphicsWindow):
    """ Class to create windows that respond to key press events.

    This code is from: https://stackoverflow.com/questions/40423999/pyqtgraph-where-to-find-signal-for-key-preses
    """
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)
