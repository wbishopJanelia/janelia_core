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


def is_standard_slice(s: slice) -> bool:
    """ Returns true if a slice has non-negative start and stop values.

    Accepts either a single slice object or a sequence of slice objects.

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


def is_fully_specified_slice(s: slice) -> bool:
    """ Returns true if slice has non non-negative, non-None start and stop values.

    Accepts either a single slice object or a sequence of slice objects.

    Args:
        s: A single slice object or sequence of slice objects.  If a sequence, this
        function will return true only if all slice objects have non-negative, non-None start and stop values.
    """

    def check_slice(s_i):
        return (s_i.start is not None) and (s_i.stop is not None) and (s_i.start >= 0) and (s_i.stop >= 0)

    if isinstance(s, slice):
        return check_slice(s)
    else:
        for s_i in s:
            passes = check_slice(s_i)
            if not passes:
                break
        return passes


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


def select_subslice(s1: slice, s2: slice) -> slice:
    """ Selects a smaller portion of a slice.

    Accepts either a single slice or a sequence of slices.

    This function expects slices to have start and stop values which are not None and non-negative.

    A subslice of a larger slice, s1, is specified by a second slice, s2.  Specifically, s2.start and s2.stop
    give the start and stop of the subslice.  s2.start and s2.stop specify the start and stop relative to s1.start.
    For example, if s1.start=5, s2.start=0, s2.stop = 2 then the returned subslice would have a start of 5 and a stop
    of 7. For simplicity, this function currently assumes s2.step = 1 but makes no assumption s1.step.  The step of
    the returned subslice will be equal to s1.step.

    When s1 and s2 are sequences of slices, s2[i] specifies the subslice for s1[i]

    Args:
        s1: The slice to select from.  Can also be a tuple of slices.

        s2: The portion of the slice to select.  Can also be a tuple of slices. The step size of this slice must
        be 1.

    Returns:
        subslice: The returned subslice.

    Raises:
        ValueError: If any slice in s1 or s2 is not a fully specified slice
        ValueError: If the step size for any s2 slice is not 1.
        ValueError: If the subslice specified by any s2 slice is not contained within the corresponding s1 slice.

    """

    if (not is_fully_specified_slice(s1)) or (not is_fully_specified_slice(s2)):
        raise(ValueError('Slices must have start and stop values which are not None and non-negative.'))

    return_tuple = True
    if type(s1) is slice:
        s1 = (s1,)
        s2 = (s2,)
        return_tuple = False

    for s2_i in s2:
        if (s2_i.step is not None) and (s2_i.step != 1):
            raise(ValueError('s2.step must be 1'))

    n_slices = len(s1)

    subslice = [None]*n_slices
    for i in range(n_slices):
        new_start = s1[i].start + s2[i].start
        new_stop = s1[i].start + s2[i].stop

        if (new_start > s1[i].stop) or (new_stop > s1[i].stop):
            raise(ValueError('Requested subslice is not contained within s1'))

        subslice[i] = slice(new_start, new_stop, s1[i].step)

    if not return_tuple:
        subslice = subslice[0]
    else:
        subslice = tuple(subslice)

    return subslice


def int_to_arb_base(base_10_vl: np.ndarray, max_digit_vls: Sequence[int]) -> np.ndarray:
    """ This is a function to convert non-zero integers to an arbitrary base.

    The base can be arbitrary in that each digit can take on a different number of values.

    Args:

        base_10_vl: The values to convert

        max_digit_vls: The length of max_digit_vls gives the length of the output representation.  max_digit_vls[i]
        gives the max value that digit i can take on.  All digit values start at 0.

    Returns:
        rep: The values in the converted format.  The least significant digit is at location 0. rep[i,:] is the
        converted value for base_10_vl[i].

    Raises:
        ValueError: If the dtype of base_10_vl is not int

        ValueError: If base_10_vl contains negative values

        ValueError: If any value in base_10_vl is too larger to represent in the base given
    """

    n_vls = len(base_10_vl)
    n_digits = len(max_digit_vls)

    cum_digit_prods = np.cumprod(max_digit_vls + 1)
    max_poss_vl = cum_digit_prods[-1] - 1

    if base_10_vl.dtype != np.int:
        raise(ValueError('Input array must be interger array.'))
    if any(base_10_vl < 0):
        raise(ValueError('Values to convert must be non-negative.'))
    if any(base_10_vl > max_poss_vl):
        raise(ValueError('One or more values is too large to represent in the specified base.'))

    res = base_10_vl

    rep = np.zeros([n_vls, n_digits], dtype=np.int)
    for d_i in range(n_digits-1, -1, -1):
        if d_i == 0:
            cur_digit_vl = 1
        else:
            cur_digit_vl = cum_digit_prods[d_i-1]

        cur_digit = np.floor(res/cur_digit_vl).astype('int')
        res -= cur_digit*cur_digit_vl
        rep[:, d_i] = cur_digit

    return rep










