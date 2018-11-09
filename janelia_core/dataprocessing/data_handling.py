""" Tools for moving and copying image files.

    William Bishop
    bishopw@hhmi.org
"""

import os
import pathlib

import pyspark


def split_images(image_files: list, base_split_folder: pathlib.Path, split_inds: list,
                 split_labels: list, sc: pyspark.SparkContext):
    """ Reads in a list of images and splits them, saving new images.

    This function is designed with the idea of splitting multi-color images that have been saved together.

    Args:
        image_files: list of image files as pathlib.Path objects

        base_split_folder: the folder to put the split data into.  If this exists, this function will throw an error
        (so users don't accidently overwrite previously split images.)

        split_inds: a list.  Each entry contains a list of indices to include in a split.

        split_labels: The labels to associate with each split.

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
        pass