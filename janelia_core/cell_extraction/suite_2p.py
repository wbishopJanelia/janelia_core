""" Helper functions for working with suite2p.

    William Bishop
    bishopw@hhmi.org
"""

import pathlib

import numpy as np
from suite2p.run_s2p import run_s2p

import janelia_core.dataprocessing.dataset
from janelia_core.fileio.keller_lab import write_planes_to_file


def run_suite2p_on_single_plane(plane_folder: pathlib.Path, ops: dict):
    """ Runs suite 2p on a single plane of multi-plane imaging data.

    This function is useful for running suite2p in parallel on multiple planes of data.

    Note: temporary suite2p results will be saved in the plane_folder.

    Args:
        plane_folder: A folder holding .h5 files for the plane at each point in time.

        ops: Structure of options to pass to suite2p.

    """

    plane_files = list(plane_folder.glob('*.h5'))
    if len(plane_files) == 0:
        raise(RuntimeError('Unable to find any .h5 files in ' + str(plane_folder)))

    ops['save_path0'] = str(plane_folder)
    db = {
        'h5py': str(plane_files[0]), # a single h5 file path as a string
        'h5py_key': 'data',
        'look_one_level_down': True,  # Even though all files are at the same level, setting this to true tells
        # suite2p to look for multiple .h5 files (vs. one file which contains all planes)
        'data_path': [],  # a list of folders with tiffs (empty since we are not using tiffs)
        'subfolders': [],  # empty since all files are on the same level
        'fast_disk': str(plane_folder),
    }

    run_s2p(ops=ops,db=db)


def default_suite2p_ops() -> dict:
    """ Returns a set of default suite2p options.

    These were modified from the suite2p github page: https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_batch.ipynb

    Returns: A dict() of default suite2p options.
    """

    ops = {
        'fast_disk': [],  # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
        'save_path0': [],  # stores results, defaults to first item in data_path
        'delete_bin': False,  # whether to delete binary file after processing
        # main settings
        'nplanes': 1,  # each tiff has these many planes in sequence
        'nchannels': 1,  # each tiff has these many channels per plane
        'functional_chan': 1,  # this channel is used to extract functional ROIs (1-based)
        'diameter': 12, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12]
        'tau': 1.,  # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)
        # output settings
        'save_mat': False,  # whether to save output as matlab files
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 0,  # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1,  # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True,  # whether to register data
        'nimg_init': 200,  # subsampled frames for finding reference image
        'batch_size': 200,  # number of frames per batch
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan': 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,  # whether to save registered tiffs
        'subpixel': 10,  # precision of subpixel registration (1/subpixel steps)
        # cell detection settings
        'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000,  # max number of binned frames for the SVD
        'nsvd_for_roi': 1000,  # max number of SVD components to keep for ROI detection
        'max_iterations': 20,  # maximum number of iterations to do cell detection
        'ratio_neuropil': 6.,  # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3,  # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1.,  # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1.,  # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75,  # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf,  # maximum neuropil radius
        'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
        # deconvolution settings
        'baseline': 'maximin',  # baselining mode
        'win_baseline': 60.,  # window for maximin
        'sig_baseline': 10.,  # smoothing constant for gaussian filter
        'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
    }

    return ops