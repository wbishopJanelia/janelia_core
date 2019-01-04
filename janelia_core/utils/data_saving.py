""" Contains tools for saving research results.

The tools here enable results to be saved and archived for later retrieval.

    William Bishop
    bishopw@hhmi.org
"""

import datetime

def append_ts(filename: str) -> str:
    """ Appends a time stamp to a string.

    The primary use case for this file is taking a file name and appending a time stamp to
    it.  This is useful when multiple analyses are run and we want to save the results of
    each with a unique name.
    """
    return (filename + '_' +  '{:%Y-%m-%d_%H:%M:%S}').format(datetime.datetime.now())
