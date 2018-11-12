""" Tools for moving and copying image files.

    William Bishop
    bishopw@hhmi.org
"""

import os
import pathlib

import h5py
import pyspark

from janelia_core.fileio.exp_reader import read_img_file


def split_images(image_files: list, base_split_folder: pathlib.Path, split_labels: list,
                 split_inds: list, h5_data_group: str = 'default', sc: pyspark.SparkContext = None):
    """ Reads in a list of images and splits them, saving new images.

    This function is designed with the idea of splitting multi-color images that have been saved together.
    It will save split data in .h5 files.

    Args:
        image_files: list of image files as pathlib.Path objects

        base_split_folder: the folder to put the split data into.  If this exists, this function will throw an error
        (so users don't accidently overwrite previously split images.)

        split_labels: The labels to associate with each split.

        split_inds: a list.  Each entry contains is itself with start and end indices for a split. A value of -1
        indicates the last entry of a dimension.

        h5_data_group: The name of the h5 datagroup to save the split data in for each split file

        sc: If provided, spark will be used to speed up computation.
    """

    # Make split folder
    if os.path.exists(base_split_folder):
        raise(IOError('The split folder already exists: ' + str(base_split_folder)))

    if len(split_inds) != len(split_labels):
        raise(RuntimeError('split_inds and split_labels must be the same length.'))

    split_folders = []
    for sl in split_labels:
        split_folder = base_split_folder / sl
        split_folders.append(split_folder)
        os.makedirs(split_folder)

    def split_file(f):
        full_images = read_img_file(f)

        for si, sl in enumerate(split_labels):
            split_image = full_images[:, split_inds[si][0]:split_inds[si][1], :]
            split_file_path = split_folders[si] / (f.stem + '_' + sl + '.h5')
            with h5py.File(split_file_path, 'w') as sf:
                dataset = sf.create_dataset(h5_data_group, split_image.shape, split_image.dtype, split_image)

    if sc is None:
        for f in image_files:
            split_file(f)
    else:
        sc.parallelize(image_files).foreach(split_file)
