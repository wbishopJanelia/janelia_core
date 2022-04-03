""" Utilities for working with file systems. """

import os
import os.path
import pathlib
from typing import List, Union


def get_immediate_subfolders(base_folder: Union[str, pathlib.Path]) -> List[str]:
    """ Gets the immediate subfolders  of a base folder.

    Args:

        base_folder: The folder to search for subfolders under.

    Returns:

        subfolders: A list of the names of subfolders.
    """
    contents = os.listdir(base_folder)
    contents = [c for c in contents if os.path.isdir(pathlib.Path(base_folder) / c)]
    return contents

