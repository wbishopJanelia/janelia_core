""" Basic math functions.

    William Bishop
    bishopw@hhmi.org
"""

from typing import Sequence
from typing import Union
import re

import numpy as np


def combine_slices(slices: Sequence[slice]) -> Sequence[slice]:
    """ Combines multiple simple slices into a potentially smaller number of simple slices.

    This function identifies overlap in slices and then combines overlapping slices into a single slice. This function
    will also get rid of slices which index nothing (have the same start and stop).

    Args:
        slices: A sequence of slice objects.  All slice objects must be simple (see is_simple_slice).

    Returns:
        c_slices: A sequence of combined slices.

    Raises:
        ValueError: If all slices are not simple.
    """

    # Handle the case of an empty list first
    if len(slices) == 0:
        return slices

    if not is_simple_slice(slices):
        raise(ValueError('All slices must be simple.'))

    min_start = np.min([s.start for s in slices])
    max_stop = np.max([s.stop for s in slices])

    shifted_slices = [slice(s.start-min_start, s.stop-min_start) for s in slices]

    bin_array = np.zeros(max_stop-min_start, dtype=np.bool)
    for s in shifted_slices:
        bin_array[s] = True

    shifted_c_slices = find_binary_runs(bin_array)
    c_slices = [slice(s.start+min_start, s.stop+min_start, 1) for s in shifted_c_slices]
    return c_slices


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


def find_binary_runs(seq: np.ndarray):
    """ Finds runs of contiguous True values in a 1-d numpy array.

    Inputs:
        seq: Array of binary values.

    Returns:
        slices: slices[i] contains a slice object for a contiguous portion of seq with all True values.

    Raises:
        ValueError: If seq is not a 1-d array.
    """

    if len(seq.shape) != 1:
        raise(RuntimeError('seq must be a 1-d numpy array.'))

    seq_b = bytearray(seq)
    matched_seqs = re.finditer(b'\x01' + b'+', seq_b)

    slices = [slice(m.span()[0], m.span()[1]) for m in matched_seqs]

    return slices


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


def is_fully_specified_slice(s: Union[slice, Sequence[slice]]) -> bool:
    """ Returns true if slice has non non-negative, non-None start and stop values.

    Accepts either a single slice object or a sequence of slice objects.

    Args:
        s: A single slice object or sequence of slice objects.  If a sequence, this
        function will return true only if all slice objects have non-negative, non-None start and stop values.

    Returns:
        is_fully_specified: True if all slice objects are fully specified
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


def is_simple_slice(s: Union[slice, Sequence[slice]]) -> bool:
    """ Returns true if a slice is fully specified and has a step size of 1 or None; otherwise returns false.

     Args:
        s: A single slice object or sequence of slice objects.  If a sequence, this
        function will return true only if all slice objects have non-negative, non-None start and stop values.

    Returns:
        is_simple: True if all slice objects are simple
     """

    def check_slice(s_i):
        if not is_fully_specified_slice(s_i):
            return False
        elif not (s_i.step == 1 or s_i.step is None):
            return False
        else:
            return True

    if isinstance(s, slice):
        return check_slice(s)
    else:
        for s_i in s:
            passes = check_slice(s_i)
            if not passes:
                break
        return passes


def is_standard_slice(s: Union[slice, Sequence[slice]]) -> bool:
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


def list_grid_pts(grid_limits: np.ndarray, n_pts_per_dim: Sequence) -> np.ndarray:
    """
    Generates a list of points filling a multi-dimensional grid.

    This function will generate a specified number of points in a range for a set of N-dimensions.
    It will then return a set of N-dimensional points formed from the cartesian product of all the
    points along each dimension.

    This function serves as a wrapper around numpy.meshgrid, essentially just repackaging
    the generated points into a single list.

    Args:

        grid_limits: grid_limits[i, :] are the limits of the grid for dimension i. Points for
        this dimension will *include* the end points of the grid.

        n_pts_per_dim[i] are the number of points to generate along dimension i. Must be greater than
        1.

    Returns:
        The list of returned points of shape n_pts*N

    Raises:
        ValueError: If n_pts_per_dim is less than 1 for any dimension.

    """

    if np.any(np.asarray(n_pts_per_dim) < 1):
        raise(ValueError('Must specify at least 2 points per dimension.'))

    n_dims = len(n_pts_per_dim)
    dim_pts = [np.linspace(start=grid_limits[d, 0], stop=grid_limits[d, 1], num=n_pts_per_dim[d])
               for d in range(n_dims)]
    m_pts = np.meshgrid(*dim_pts, indexing='ij')
    m_pts = [np.reshape(vls, vls.size) for vls in m_pts]
    return np.stack(m_pts).transpose()


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


def optimal_orthonormal_transform(m_0: np.ndarray, m_1: np.ndarray) -> np.ndarray:
    """ Learns an optimal orthonormal transformation to transform one matrix to another.

    We learn a matrix o^* = argmin_{o: oo^T = I} ||m_0 - m_1 o||_2.

    Args:
        m_0: The matrix to transform to

        m_1: The matrix to transform from

    Returns:
        o: The optimal orthonormal matrix.

    """

    s = np.matmul(m_0.transpose(), m_1)
    u, _, v_transpose = np.linalg.svd(s)
    return (np.matmul(u, v_transpose)).transpose()


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








