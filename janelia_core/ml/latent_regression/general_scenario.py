""" This provides high-level tools for working with latent regression models of multiple groups of input
driving multiple groups of output.

"""

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch.nn

from janelia_core.ml.datasets import TimeSeriesBatch
from janelia_core.ml.extra_torch_modules import BiasAndPositiveScale
from janelia_core.ml.latent_regression.group_maps import GroupLinearTransform
from janelia_core.ml.latent_regression.subject_models import SharedMLatentRegModel
from janelia_core.ml.latent_regression.vi import SubjectVICollection


class GeneralInputOutputScenario():
    """ Class for working with latent regression models of multiple groups of input and output across multiple subjects.

    In particular, we assume all subjects:

        1) Have the same input groups (but the number of variables in each group can vary from subject to
        subject)

        2) Have the same output groups (but again, the number of variables in each group can vary from subject to
        subject).

        3) We also assume that m-module (the mapping from low-dimenaional input to low-dimensional output) has two
        components for each subject: a component that is shared between all subjects and a component that is unique to
        each subject, where the subject-unique component is applied to the low-dimensioanal projections of inputs and
        it's output is passed to the shared-component.

    Note: When creating the initial scenario, subjects are entered in a particular order. (e.g., the number of variables
    in input and output groups for each subject).  This ordering corresponds to their index, s_i, which is used in
    multiple functions.


    """

    def __init__(self, input_dims: np.ndarray, output_dims: np.ndarray, n_input_modes: Sequence[int],
                 n_output_modes: Sequence[int], shared_m_core: torch.nn.Module,
                 use_scales: bool = True, use_offsets: bool = True, direct_pairs: Sequence[tuple] = None,
                 p_mode_point_ests: Union[bool, Sequence[bool]] = False,
                 u_mode_point_ests: Union[bool, Sequence[bool]]= False,
                 scales_point_ests: Union[bool, Sequence[bool]] = False,
                 offsets_point_ests: Union[bool, Sequence[bool]] = False,
                 direct_mappings_point_ests: Union[bool, Sequence[bool]] = False,
                 psi_point_ests: Union[bool, Sequence[bool]] = False,
                 fixed_p_modes: Union[None, Sequence[tuple]] = None, fixed_u_modes: Union[None, Sequence[tuple]] = None,
                 fixed_scales: Union[None, Sequence[tuple]] = None, fixed_offsets: Union[None, Sequence[tuple]] = None,
                 fixed_direct_mappings: Union[None, Sequence[tuple]] = None,
                 fixed_psi: Union[None, Sequence[tuple]] = None):

        """ Creates a GenInputOutputScenario object.

        The user had the option estimate point estimates over parameters or full posterior distributions. For a given
        parameter, the user can specify what he or she would like to do by using the appropriate '*_point_ests'
        argument. This argument can be specified in two ways.  Providing a single boolean value indicates point
        estimates should or should not be used for all groups for that parameter.  Alternatively, the user can specify
        a sequence of boolean values indicatinb which groups point estimates should be used for.

        Args:

            input_dims: Dimensions of input groups of variables for each subject.  input_dims[s, g] is the number of
            input variables in group g for subject s

            ouput_dims: Dimensions of output groups of variables for each subject.  output_dims[s, h] is the number of
            output variables in group h for subject s

            n_input_modes: n_input_modes[g] is the number of modes for input group g

            n_output_modes: n_output_modes[h] is the number of modes for output group h

            shared_m_core: The portion of the m-module that is shared between subjects.  This should be a module which
            receives input in the form of a sequence the same length as the number of input groups and produces output
            which is also a sequence with a length equal to the number of output groups.  The dimensionality of the
            input and output tensors should match the number of modes of the respective input and output groups.

            use_scales: True if models should including scaling of output

            use_offsets: True if models should include offsets applied to output

            direct_pairs: Pairs of input and output groups which have direction connections.

            p_mode_point_ests: Indicates if point estimtates should be used for the p modes.  See note above on
            specifying the form of this argument.

            u_mode_point_ests: Indicates if point estimtates should be used for the u modes.  See note above on
            specifying the form of this argument.

            scales_point_ests: Indicates if point estimtates should be used for the scales.  See note above on
            specifying the form of this argument.

            offsets_point_ests: Indicates if point estimtates should be used for the offsets.  See note above on
            specifying the form of this argument.

            direct_mappings_point_ests: Indicates if point estimtates should be used for the direct mappings.
            If provided as a list, the i^th entry indicates if the the direct mapping for the i^th entry in
            direct_pairs should be estimated with a point estimate.

            psi_point_ests: Indicates if point estimtates should be used for the psi parameters.  See note above on
            specifying the form of this argument.

            fixed_p_modes: A sequence of tuples of the form (g, p_g) where g is the index of an input group and
            p_g is a tensor that should be used as fixed (non-learnable) modes for this input group across all subjects.

            fixed_u_modes: A sequence of tuples specifying fixed output modes in the same manner as
            fixed_p_modes.

            fixed_scales: A sequence of scales of the form (h, t_h) where h is the index of an output group and
            t_h is a tensor of scales that should be used as fixed (non-learnable) scale parameter for that output
            group across all subjects.

            fixed_offsets: A sequence of tuples specifying fixed offset parameters, in the same form as fixed_scales.

            fixed_direct_mappings: A sequence of tuples specifying fixed direct pair mappings of the form (i, t_i)
            where i is the index into direct pairs giving the pair the mapping is for and t_i is the fixed mapping
            that should be used.

            fixed_psi: A sequence of tuples specifying fixed psi parameters, in the same form as fixed scales.

        """

        def _format_pe(vl, n_grps):
            if isinstance(vl, bool):
                return [vl]*n_grps
            else:
                return vl

        n_input_grps = len(n_input_modes)
        n_output_grps = len(n_output_modes)

        self.n_input_grps = n_input_grps
        self.n_output_grps = n_output_grps
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.n_input_modes = n_input_modes
        self.n_output_modes = n_output_modes
        self.shared_m_core = shared_m_core
        self.use_scales = use_scales
        self.use_offsets = use_offsets
        self.direct_pairs = direct_pairs
        self.p_mode_points_ests = _format_pe(p_mode_point_ests, n_input_grps)
        self.u_mode_points_ests = _format_pe(u_mode_point_ests, n_output_grps)
        self.scales_points_ests = _format_pe(scales_point_ests, n_output_grps)
        self.offsets_points_ests = _format_pe(offsets_point_ests, n_output_grps)
        self.direct_mappings_points_ests = _format_pe(direct_mappings_point_ests, n_output_grps)
        self.psi_points_ests = _format_pe(psi_point_ests, n_output_grps)
        self.fixed_p_modes = fixed_p_modes
        self.fixed_u_modes = fixed_u_modes
        self.fixed_scales = fixed_scales
        self.fixed_offsets = fixed_offsets
        self.fixed_direct_mappings = fixed_direct_mappings
        self.fixed_psi = fixed_psi

    def gen_subj_mdl(self, s_i: int, specific_s: Sequence[torch.nn.Module],
                     specific_m: torch.nn.Module = None, w_gain: float = 1.0, sc_std: float = .01, dm_std: float = 1.0,
                     noise_range: Sequence[float] = [.1, .2]) -> SharedMLatentRegModel:

        """ Generates a subject model for a particular subject.

        Args:

            s_i: The index of the subject to generate a subject model for.

            specific_s: specific_s[h] is the module to be applied to the output group h values after they have been
            projected back into the high-dimensional space (i.e., multiplied by u_h).

            specific_m: An optional module which can apply subject specific transformations to the low-dimensional
            projections of the input data before passing it to the shared portion of the m-module. This can be useful,
            for example, if wanting to account for scales and shifts in the projected data among subjects. This should
            be a module which accepts a sequence of tensors, each tensor corresponding to the projected data from one
            output group and outputs a sequence of tensors, each tensor corresponding to the transformed data for one
            output group.  If this is None, no subject specific transformation will be included.

            w_gain, sc_std, dm_std, noise_range: Value to provide to SharedLatentRegModel.  See documentation for
            that object.

        Returns:

            mdl: The requested subject model
        """

        def _gen_assign_vls(used: bool, point_estimates: Sequence[bool],
                            fixed_tuples: Sequence[tuple]):
            if not used:
                return False
            else:
                # We need to assign values either because we are using point estimates for a parameter of they are
                # fixed
                assign_vls = point_estimates
                if fixed_tuples is not None:
                    for g, _ in fixed_tuples:
                        assign_vls[g] = True
                return assign_vls

        assign_scales = _gen_assign_vls(self.use_scales, self.scales_points_ests, self.fixed_scales)
        assign_offsets = _gen_assign_vls(self.use_offsets, self.offsets_points_ests, self.fixed_offsets)
        assign_direct_mappings = _gen_assign_vls(self.direct_pairs is not None, self.direct_mappings_points_ests,
                                                 self.fixed_direct_mappings)
        assign_p_modes = _gen_assign_vls(True, self.p_mode_points_ests, self.fixed_p_modes)
        assign_u_modes = _gen_assign_vls(True, self.u_mode_points_ests, self.fixed_u_modes)
        assign_psi = _gen_assign_vls(True, self.psi_points_ests, self.fixed_psi)

        mdl = SharedMLatentRegModel(d_in=self.input_dims[s_i, :], d_out=self.output_dims[s_i, :],
                                    d_proj=self.n_input_modes, d_trans=self.n_output_modes,
                                    specific_m=specific_m, shared_m=self.shared_m_core,
                                    s=specific_s, use_scales=self.use_scales, assign_scales=assign_scales,
                                    use_offsets=self.use_offsets, assign_offsets=assign_offsets,
                                    direct_pairs=self.direct_pairs,
                                    assign_direct_pair_mappings=assign_direct_mappings,
                                    assign_p_modes=assign_p_modes, assign_u_modes=assign_u_modes,
                                    assign_psi=assign_psi, w_gain=w_gain, sc_std=sc_std, dm_std=dm_std,
                                    noise_range=noise_range)

        # Set fixed values if we need to
        def _set_fixed_vls(params: Sequence[torch.Tensor], trainable: Sequence[bool], fixed_vls):
            if fixed_vls is not None:
                for i, t_i in fixed_vls:
                    params[i].data = t_i
                    trainable[i] = False

        _set_fixed_vls(mdl.p, mdl.p_trainable, self.fixed_p_modes)
        _set_fixed_vls(mdl.u, mdl.u_trainable, self.fixed_u_modes)
        if self.use_scales:
            _set_fixed_vls(mdl.scales, mdl.scales_trainable, self.fixed_scales)
        if self.use_offsets:
            _set_fixed_vls(mdl.offsets, mdl.offsets_trainable, self.fixed_offsets)
        if self.direct_pairs is not None:
            _set_fixed_vls(mdl.direct_mappings, mdl.direct_mappings_trainable, self.fixed_direct_mappings)
        _set_fixed_vls(mdl.psi, mdl.psi_trainable, self.fixed_psi)

        return mdl

    def gen_linear_specific_m(self, scale_mn=0.001, scale_std=.00001,
                                   offset_mn=0.0, offset_std=.00001) -> torch.nn.Module:
        """ Generates a subject-specific component of the m-module for non-negative scales and offsets.

        Args:
            scale_mn, scale_std, offset_mn, offset_std: the mean and standard deviaton of the normal distribution when
            drawing random initial values for the scale and offset.

        Returns:
            m: The subject-specific component.

        """
        return GroupLinearTransform(d=self.n_input_modes, nonnegative_scale=True, v_mn=scale_mn, v_std=scale_std,
                                    o_mn=offset_mn, o_std=offset_std)

    def gen_linear_specific_s(self, s_i: int, scale_mn=1.0, scale_std=.01, offset_mn=0.0,
                                   offset_std=.000001) -> List[torch.nn.Module]:
        """ Generates subject-specific sequence of s-modules which apply an element-wise non-negative scale and bias.

        Args:

            s_i: The subject to generate the s-modules for.

            scale_mn, scale_std, offset_mn, offset_std: the mean and standard deviaton of the normal distribution when
            drawing random initial values for the scale and offset.
        """
        return [BiasAndPositiveScale(d=d_h, o_init_mn=offset_mn, o_init_std=offset_std,
                                     w_init_mn=scale_mn, w_init_std=scale_std) for d_h in self.output_dims[s_i, :]]


def calc_projs_given_post_modes(s_vi_collection: SubjectVICollection, input_modules: torch.nn.ModuleList,
                                data: Sequence[TimeSeriesBatch],
                                apply_subj_specific_m: bool = False) -> Tuple[Sequence, Sequence]:
    """ Calculates values of projected data, given posterior distributions, over a model's modes.

    This function will project data using the means of posterior distributions over modes and will calculate
    projected values immediately after they are projected to the low-d space.

    Args:

        s_vi_collection: The vi collection to produce projections for

        data: Data to project. x[g] is input data for group g

        input_modules: Input modules to apply to input before projecting

        apply_subj_specific_m: True if subject-specific m components of the m-module should be applied after the
        projections

    Returns:

        p_projs: The projections onto the p-modes (potentially after the subject-specific m module has been applied)

        u_projs: The projections after having been transformed through the m-module.

    """

    data = [data.data[i_g][data.i_x, :] for i_g in s_vi_collection.input_grps]

    # Apply input modules
    data = [x_g if i_m is None else i_m(x_g) for i_m, x_g in zip(input_modules, data)]

    # Get the posterior p modes
    if s_vi_collection.p_dists is None:
        q_p_modes = None
    else:
        q_p_modes = [d(s_vi_collection.props[s_vi_collection.p_props[g]]) if d is not None
                     else None
                     for g, d in enumerate(s_vi_collection.p_dists)]

    if apply_subj_specific_m:
        p_projs = s_vi_collection.s_mdl.p_project(data, q_p_modes, apply_specific_m=apply_subj_specific_m)
    else:
        p_projs = s_vi_collection.s_mdl.p_project(data, q_p_modes)

    u_projs = s_vi_collection.s_mdl.m(s_vi_collection.s_mdl.p_project(data, q_p_modes))

    return p_projs, u_projs
