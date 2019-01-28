""" Basic math functions.

    William Bishop
    bishopw@hhmi.org
"""

import numpy as np


def l_th(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ Thresholds an array with a lower bound.

    Values less than the threshold value are set to the threshold value.

    Args:
        a: the array to threshold

        t: the threshold to use

    Returns: The thresholded array
    """
    a[np.where(a < t)] = t
    return a


def u_th(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ Thresholds an array with an upper bound.

    Values greater than the threshold value are set to the threshold value.

    Args:
        a: the array to threshold

        t: the threshold to use

    Returns: The thresholded array
    """
    a[np.where(a > t)] = t
    return a


def divide_into_nearly_equal_parts(n, k) -> np.ndarray:
    """ Produces k nearly equal integer values which sum to n.

    All values will differ by no more than 1.

    Args:
        n: The number to divide

        k: The number to divide n by.

    Returns:
        An array of length k of values that sum to n.  Larger values will be in lower indices.
    """

    if k > n:
        base_vl = 0
    else:
        base_vl = n//k

    vls = np.full(k, base_vl)
    rem = int(n - (base_vl*k))

    for i in range(rem):
        vls[i] += 1

    return vls


def find_first_before(a: np.ndarray, ind: int) -> int:
    """ Given a logical array, finds first true value before or at a given index.

    Args:
        a: The logical array to search

        ind: The index to start the search at.

    Returns:
        first_ind: The index the first true value occurs at. If no match is found, returns None.

    Raises:
        ValueError: If ind is negative or out of the range of the array.

    """

    len_a = len(a)

    if ind > len_a - 1:
        raise(RuntimeError('Index ' + str(ind) +
                           ' is out of range for an array with length ' + str(len_a) + '.'))
    if ind < 0:
        raise(RuntimeError('Index must be positive.'))

    a = a[0:ind+1]
    inds = np.where(a == 1)[0]
    if inds.size == 0:
        return None
    else:
        return inds[-1]


def find_first_after(a: np.ndarray, ind: int) -> int:
    """ Given a logical array, finds first true value after or at a given index.

    Args:
        a: The logical array to search

        ind: The index to start the search at.

    Returns:
        first_ind: The index the first true value occurs at. If no match is found, returns None.

    Raises:
        ValueError: If ind is negative or out of the range of the array.

    """

    len_a = len(a)

    if ind > len_a - 1:
        raise(RuntimeError('Index ' + str(ind) +
                           ' is out of range for an array with length ' + str(len_a) + '.'))
    if ind < 0:
        raise(RuntimeError('Index must be positive.'))

    a = a[ind:len_a]
    inds = np.where(a == 1)[0]
    if inds.size == 0:
        return None
    else:
        return inds[0] + ind


