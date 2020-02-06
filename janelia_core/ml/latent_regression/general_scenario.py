""" This provides high-level tools for working with latent regression models of multiple groups of input
driving multiple groups of output.

"""

from typing import Sequence

import numpy as np


class GenInputOutputScenario():
    """ Class for working with latent regression models of multiple groups of input and output across multiple subjects.

    In particular, we assume all subjects:

        1) Receives input the same input groups (but the number of variables in each group can vary from subject to
        subject)

        2) The input from each group, is projected into some


    """

    def __init__(self, input_dims: np.ndarray, output_dims: np.ndarray, n_input_modes: Sequence[int],
                 n_output_modes: Sequence[int], fixed_input_modes: Sequence(tuple) = None,
                 fixed_output_modes: Sequence(tuple) = None):
        """ Creates a GenInputOutputScenario object.

        Args:
            input_dims: Dimensions of input groups of variables for each subject.  input_dims[s, g] is the number of
            input variables in group g for subject s

            ouput_dims: Dimensions of output groups of variables for each subject.  input_dims[s, h] is the number of
            output variables in group h for subject s

            n_input_modes: n_input_modes[g] is the number of modes for input group g

            n_output_modes: n_output_modes[h] is the number of modes for output group h

            fixed_input_modes: A sequence of tuples of the form (g, p_g) where g is the index of an input group and
            p_g is a tensor that should be used as fixed (non-learnable) modes for this input group across all subjects

            fixed_output_modes: A sequence of tuples specifying fixed output modes in the same manner as
            fixed_input_modes


        """
        pass

    def gen_subj_mdl(self):
        """ Generates a subject model for a particular subject. """
        pass






