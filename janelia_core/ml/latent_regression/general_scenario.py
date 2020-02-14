""" This provides high-level tools for working with latent regression models of multiple groups of input
driving multiple groups of output.

"""

from typing import List, Sequence

import numpy as np
import torch.nn

from janelia_core.ml.datasets import TimeSeriesBatch
from janelia_core.ml.extra_torch_modules import BiasAndScale
from janelia_core.ml.latent_regression.group_maps import GroupLinearTransform
from janelia_core.ml.latent_regression.subject_models import LatentRegModel
from janelia_core.ml.latent_regression.vi import SubjectVICollection


class GeneralInputOutputScenario():
    """ Class for working with latent regression models of multiple groups of input and output across multiple subjects.

    In particular, we assume all subjects:

        1) Have the same input groups (but the number of variables in each group can vary from subject to
        subject)

    Note: When creating the initial scenario, subjects are entered in a particular order. (e.g., the number of variables
    in input and output groups for each subject).  This ordering corresponds to their index, s_i, which is used in
    multiple functions.


    """

    def __init__(self, input_dims: np.ndarray, output_dims: np.ndarray, n_input_modes: Sequence[int],
                 n_output_modes: Sequence[int], shared_m_core: torch.nn.Module,
                 fixed_input_modes: Sequence[tuple] = None,
                 fixed_output_modes: Sequence[tuple] = None, direct_pairs: Sequence[tuple] = None):

        """ Creates a GenInputOutputScenario object.

        Args:

            input_dims: Dimensions of input groups of variables for each subject.  input_dims[s, g] is the number of
            input variables in group g for subject s

            ouput_dims: Dimensions of output groups of variables for each subject.  input_dims[s, h] is the number of
            output variables in group h for subject s

            n_input_modes: n_input_modes[g] is the number of modes for input group g

            n_output_modes: n_output_modes[h] is the number of modes for output group h

            shared_m_core: The portion of the m-module that is shared between subjects.  This should be a module which
            receives input in the form of a sequence the same length as the number of input groups and produces output
            which is also a sequence with a length equal to the number of output groups.  The dimensionality of the
            input and output tensors should match the number of modes of the respective input and output groups.

            fixed_input_modes: A sequence of tuples of the form (g, p_g) where g is the index of an input group and
            p_g is a tensor that should be used as fixed (non-learnable) modes for this input group across all subjects.

            fixed_output_modes: A sequence of tuples specifying fixed output modes in the same manner as
            fixed_input_modes.

            direct_pairs: Pairs of input and output groups which have direction connections.

        """

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.n_input_modes = n_input_modes
        self.n_output_modes = n_output_modes
        self.shared_m_core = shared_m_core
        self.fixed_input_modes = fixed_input_modes
        self.fix_output_modes = fixed_output_modes
        self.direct_pairs = direct_pairs

    def gen_subj_mdl(self, s_i: int, specific_s: Sequence[torch.nn.Module],
                     specific_m: torch.nn.Module = None, assign_p_u: bool = True) -> LatentRegModel:

        """ Generates a subject model for a particular subject.

        Args:

            s_i: The index of the subject to generate a subject model for.

            specific_s: specific_s[h] is the module to be applied to the output group h values after they have been
            projected back into the high-dimensional space (i.e., multiplied by u_h).

            specific_m: An optional module which can apply subject specific transformations to the low-dimensional
            projections of the input data.  This can be useful, for example, if wanting to account for scales and
            shifts in the projected data among subjects. This should be a module which accepts a sequence of tensors,
            each tensor corresponding to the projected data from one output group and outputs a sequence of tensors,
            each tensor corresponding to the transformed data for one output group.  If this is None, no subject
            specific transformation will be included.

            assign_p_u: True if p and u parameters should be generated for the subject model.

        Returns:

            mdl: The requested subject model
        """

        if specific_m is None:
            m = self.shared_m_core
        else:
            m = torch.nn.Sequential(self.shared_m_core, specific_m)

        mdl = LatentRegModel(d_in=self.input_dims[s_i, :], d_out=self.output_dims[s_i, :],
                             d_proj=self.n_input_modes, d_trans=self.n_output_modes,
                             m=m, s=specific_s, direct_pairs=self.direct_pairs,
                             assign_p_u=assign_p_u)

        if assign_p_u:
            if self.fixed_input_modes is not None:
                for g, p_g in self.fixed_input_modes:
                    mdl.p[g].data = p_g
                    mdl.p_trainable[g] = False
            if self.fixed_input_modes is not None:
                for h, u_h in self.fix_output_modes:
                    mdl.u[h].data = u_h
                    mdl.u_trainable[h] = False

    def gen_subj_vi_collection(self, s_i: int, specific_s: Sequence[torch.nn.Module], p_dists: Sequence,
                               u_dists: Sequence, data: TimeSeriesBatch, props: Sequence,
                               specific_m: torch.nn.Module = None, assign_p_u: bool = True,
                               min_var:float = .001):

        s_mdl = self.gen_subj_mdl(s_i=s_i, specific_s=specific_s, specific_m=specific_m, assign_p_u=assign_p_u)

        return SubjectVICollection(s_mdl=s_mdl, p_dists=p_dists, u_dists=u_dists, data=data,
                                   input_grps=(0, 1), output_grps=(0, 2), props = [props],
                                   input_props=[0, None], output_props=[0, None],
                                   min_var=[min_var, min_var])


    def gen_linear_specific_m(self, s_i: int, scale_mn=0.0, scale_std=.01,
                                   offset_mn=0.0, offset_std=.00001) -> torch.nn.Module:
        """ Generates a subject-specific component of the m-module for scales and offsets.

        Args:
            s_i: The subject to generate the map for

            scale_mn, scale_std, offset_mn, offset_std: the mean and standard deviaton of the normal distribution when
            drawing random initial values for the scale and offset.

        Returns:
            m: The subject-specific component.

        """
        return GroupLinearTransform(d=self.n_input_modes, v_mn=scale_mn, v_std=scale_std,
                                    o_mn=offset_mn, o_std=offset_std)

    def gen_linear_specific_s(self, s_i: int, scale_mn=1.0, scale_std=.01, offset_mn=0.0,
                                   offset_std=.000001) -> List[torch.nn.Module]:
        """ Generates subject-specific sequence of s-modules.

        Args:

            s_i: The subject to generate the s-modules for.

            scale_mn, scale_std, offset_mn, offset_std: the mean and standard deviaton of the normal distribution when
            drawing random initial values for the scale and offset.
        """
        return [BiasAndScale(d=d_h, o_init_mn=offset_mn, o_init_std=offset_std,
                             w_init_mn=scale_mn, w_init_std=scale_std) for d_h in self.output_dims[s_i, :]]



