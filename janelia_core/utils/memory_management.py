""" Contains tools for working with memory. """

import os
import psutil


def get_current_memory_usage():
    """ Returns the memory used by the process this function is called from.

    Returns:
        mem: Memory used in gigabytes
    """

    BYTES_TO_GB = 1024**3

    process = psutil.Process(os.getpid())
    return process.memory_info()[0]/BYTES_TO_GB