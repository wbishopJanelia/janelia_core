""" Basic math functions.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence

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


def slice_contains(s1: slice, s2: slice):
    """ Checks if the range of slice s1 is contained in the range of slice s2.

    This function accepts either single slices or a sequence of slices for s1 and s2. If a
    sequence of slices is provided, s1[i] is interpreted as the slice for dimension i. If s1
    and s2 are sequences but of different lengths, this function will immediately return false.

    This function currently does not accept slices with negative start or stop values.

    Args:
        s1: The slice we are checking to see if it is contained by s2. Can also be
        a sequence of slices.

        s2: The slice we are checking to see if it contains s1.  Can also be a sequence
        of slices

    Raises:
        NotImplementedError: If slice objects contain negative start or stop values.
    """

    def check_single_slice(sl1, sl2):
        if (not is_standard_slice(sl1)) or (not is_standard_slice(sl2)):
            raise(NotImplementedError(('Checking for containment of slices containing negative start or stop'
                                       ' values is not currently supported.')))
        return (sl2.start <= sl1.start) & (sl2.stop >= sl1.stop)

    if isinstance(s1, slice):
        return check_single_slice(s1, s2)
    else:
        n_s1_dims = len(s1)
        if n_s1_dims != len(s2):
            return False
        else:
            for d in range(n_s1_dims):
                contained = check_single_slice(s1[d], s2[d])
                if not contained:
                    break
            return contained


def is_standard_slice(s: slice):
    """ Returns true if a slice has non-negative start and stop values.

    Accepts either a single slice object of sequence of slice objects.

    Args:
        s: A single slice object or sequence of slice objects.  If a sequence, this
        function will return true only if all slice objects are standard.

    Returns:
        is_standard: True if all slice objects are standard.
    """
    def check_slice(s_i):
        return s_i.start >= 0 and s_i.stop >= 0

    if isinstance(s, slice):
        return check_slice(s)
    else:
        for s_i in s:
            is_standard = check_slice(s_i)
            if not is_standard:
                break
        return is_standard


def nan_matrix(shape: Sequence[int], dtype=np.float):
    """ Generates a matrix of a given shape and data type initialized to nan values.

    Args:
        shape: Shape of the matrix to generate.

        dtype: Data type of the matrix to generate.

    Returns:
        The generated matrix.

    """

    m = np.empty(shape=shape, dtype=dtype)
    m.fill(np.nan)
    return m



