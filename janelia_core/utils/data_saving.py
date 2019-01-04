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

    The time stamps will be of the format _<4 digit year>_<2 digit month>_<two digit day>_<2 digit military time hour>...
                                            _<2 digit minute>_<2 digit second>_<0 padded microsecond>
    """
    return (filename + '_' + '{:%Y_%m_%d_%H_%M_%S_%f}').format(datetime.datetime.now())
