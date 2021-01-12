""" Holds modules for penalizing torch parameters. """

from abc import ABC, abstractmethod
import copy
from typing import List, Sequence

import numpy as np
import torch


class ParameterPenalizer(ABC, torch.nn.Module):
    """ An abstract class for parameter penalizers.

    The main idea behind a penalizer object is that when fitting models instead of simply applying a penalty function,
    we might want to penalize parameters with respect to a function which also has its own learnable parameters. For
    example, lets say there are a group of parameters and we want to penalize things so that the l_2 distance between
    all the parameters in that group is small. Instead of calculating the l_2 distance between all pairs of parameters
    and penalizing the sum of those distances, it might be simpler to have a penalizer with a center parameter which is
    learned and penalizing the distance of each parameter in the group to the center.

    Note one point of confusion is there are now two sets of parameters - one set is the set of parameters that we
    want to penalize and the other set is the set of internal, learnable parameters of the penalizer itself.  In
    the example above, the center would be an internal, learnable parameter of the penalizer.

    """

    def __init__(self, params: Sequence[torch.nn.Parameter]):
        """ Creates an instance of a penalizer.

        Args:
            params: The parameters to penalize over.
        """
        super().__init__()
        self.params = params

    @abstractmethod
    def check_point(self) -> dict:
        """ Returns a dictionary of parameters and values for the penalizer that should be saved in a check point.

        The idea is that for the purposes of creating a check point, we can save memory by only logging the
        important parameters and values of a penalizer.
        """
        raise(NotImplementedError())

    @abstractmethod
    def clone(self, clean: bool = True):
        """ Returns a copy of self.

        Args:
            clean: If true, attribute values that we might not want to transfer to a new object (such as record of last
            penalty value) will not be copied.

        Returns:
            obj: The new object

        """
        raise(NotImplementedError())

    @abstractmethod
    def copy_state_from(self, other):
        """ Copies the state of one penalizer to this penalizer.

        State should be the internal parameters of the penalizers as well as other internal varialbes it may keep but
        it should not include the parameters the penalizer actually penalizes.

        Args:
            other: The other penalizer to copy state from
        """
        raise(NotImplementedError())

    @abstractmethod
    def get_marked_params(self, key: str) -> List[torch.nn.Parameter]:
        """ Returns all learnable parameters marked with the key string.

        Penalizers must associate each of their internal, learnable parameters with a unique key
        (e.g., fast_learning_rate_params).  Each parameter should be associated with only one key
        (though multiple parameters can use the same key).  This function will return a list of parameters
        associated with the requested key.  If no parameters match the key an empty list should be returned.

        """
        raise(NotImplementedError())

    @abstractmethod
    def list_param_keys(self) -> List[str]:
        """ Returns a list of keys associated with internal, learnable parameters.

        Returns:
            keys: The list of keys.
        """
        raise(NotImplementedError())

    @abstractmethod
    def penalize_and_backwards(self, call_backwards: bool) -> float:
        """ Calculates a penalty over parameters and calls backwards on the penalty.

        The reason for having the penalizer call backwards is that there may be complicated situations, such as
        when parameters are spread over multiple GPUs, that we need to call backwards multiple times as we
        move things between GPUs when calculating the penalty.

        Args:
            d: The distribution to penalize

        Returns:
            penalty: The scalar penalty.  Note this is a float and not a tensor, as we assume backwards has
            been called in this function.
        """
        raise(NotImplementedError())

    @abstractmethod
    def __str__(self) -> str:
        """ Returns a string of the current state of the penalizer. """
        raise(NotImplementedError)


class ClusterPenalizer(ParameterPenalizer):
    """ This penalizer encourages clustering of parameters in tensors.

    In particular, given a set of paremters p_0, ..., p_N of arbitrary shape, the penalty computed by this object is:

        w\sum_{i=1}^N ||p_i - c||_2^2 where c is a tensor the same shape as any of the p_i parameters representing
        the center of a cluster and w is a penalty weight.

    There is only one parameter for this penalizer, which is tagged with 'fast', to indicate that in models trained
    with slow and fast learning rates, we would expect the center to be updated with the fast learning rate.
    """

    def __init__(self, params: Sequence[torch.nn.Parameter], w: float, init_ctr: torch.Tensor,
                 description:str = None, learnable_parameters: bool = True):
        """ Creates an instance of a ClusterPenalizer.

        Args:
            params: The parameters to penalize

            w: The weight to apply the penalty

            init_ctr: The initial value of c

            description: A string that will be used to identify the penalizer in the string returned
            by __str__()

            learnable_parameters: True if c should be learnable; false if it should be fixed
        """
        super().__init__(params)
        self.w = w
        self.learnable_parameters = True

        if learnable_parameters:
            self.c = torch.nn.Parameter(init_ctr)
        else:
            self.register_buffer('c', init_ctr)
        self.last_p = np.nan
        if description is not None:
            self.description = description
        else:
            self.description = 'Parameter Penalizer'

    def copy_state_from(self, other):
        """ Copies the state of another penalizer to this penalizer.

        Args:
            other: The other penalizer to copy state form.
        """
        with torch.no_grad():
            self.c.data = other.c.data
        self.last_p = other.last_p

    def clone(self, clean:bool = True):
        """ Returns a copy of self.

        Args:
            clean: If true, attribute values that we might not want to transfer to a new object (such as record of last
            penalty value) will not be copied.

        Returns:
            obj: The new object

        """

        self_copy = copy.deepcopy(self)
        if clean:
            self_copy.last_p = np.nan

        return self_copy

    def check_point(self) -> dict:
        """ Returns a check point dictionary for the penalizer.

        Returns:
            d: A dictionary with the following keys:
                c: The center of the parameter
                last_p: The value of the last penalty that was computed
        """

        c = copy.deepcopy(self.c)
        c = c.detach().cpu().numpy()

        return {'c': c, 'last_p': self.last_p}

    def get_marked_params(self, key: str) -> List[torch.nn.Parameter]:
        """ Returns marked parameters.

        The only parameter of the penalizer is the centers tensor, c, which is marked with the tag 'fast'

        Returns:
            params: A list.  If the key was fast this will hold the 'c' parameter.  If not, this list will be empty.
        """

        if key == 'fast':
            return [self.c]
        else:
            return []

    def list_param_keys(self)  -> List[str]:
        """ Returns the list of keys associated with internal, learnabke parameters. """
        return ['fast']

    def penalize_and_backwards(self, call_backwards: bool = True) -> torch.Tensor:
        """ Computes the penalty over the parameters and then calls backwards.

        Args:
            call_backwards: True if backwards should be called.

        Returns:
            penalty: The calculated penalty
        """

        penalty = 0.0
        if self.w > 0:
            for p in self.params:
                self.to(p.device)
                cur_penalty = self.w*torch.sum(torch.sum((p - self.c)**2))
                if call_backwards:
                    cur_penalty.backward()
                penalty += cur_penalty.detach().cpu().numpy()

        self.last_p = penalty
        return penalty

    def __str__(self):
        """ Returns a string with the state of the penalizer. """
        return (self.description + ' state'+
                '\n Center: ' + str(self.c.detach().cpu().numpy()) +
                '\n Last Penalty: ' + str(self.last_p))


class UnsignedClusterPenalizer(ClusterPenalizer):
    """ This is the same as the ClusterPenalizer but the penalty is computed after taking absolute values of parameters.

    In particular, given a set of paremters p_0, ..., p_N of arbitrary shape, the penalty computed by this object is:

        w\sum_{i=1}^N ||abs(p_i) - abs(c)||_2^2 where c is a tensor the same shape as any of the p_i parameters
        representing the center of a cluster and w is a penalty weight.

    """

    def __init__(self, params: Sequence[torch.nn.Parameter], w: float, init_ctr: torch.Tensor,
                 description:str = None, learnable_parameters: bool = True):
        """ Creates a new UnsignedClusterPenalizer instance.

        See __init__ for ClusterPenalizer for documenation.
        """
        super().__init__(params=params, w=w, init_ctr=init_ctr, description=description,
                         learnable_parameters=learnable_parameters)

    def penalize_and_backwards(self, call_backwards: bool = True):
        """ Computes the penalty over the parameters and then calls backwards.

        Args:
            call_backwards: True if backwards should be called.

        Returns:
            penalty: The calculated penalty
        """

        penalty = 0.0
        if self.w > 0:
            for p in self.params:
                self.to(p.device)
                cur_penalty = self.w*torch.sum(torch.sum((torch.abs(p) - torch.abs(self.c))**2))
                if call_backwards:
                    cur_penalty.backward()
                penalty += cur_penalty.detach().cpu().numpy()

        self.last_p = penalty
        return penalty


class ScalarPenalizer(ParameterPenalizer):
    """ Applies an element-wise penalty to parameters, which is the squared distance of each element from a center.

        In particular, given a set of paremters p_0, ..., p_N of arbitrary shape, the penalty computed by this object is:

        w\sum_{i=1}^N ||p_i - c||_2^2 where c the scalar center.

    There is only one parameter for this penalizer, which is tagged with 'fast', to indicate that in models trained
    with slow and fast learning rates, we would expect the center to be updated with the fast learning rate.

    """

    def __init__(self, params: Sequence[torch.nn.Parameter], w: float, init_ctr: float,
                 description:str = None, learnable_parameters: bool = True):
        """ Creates an instance of a ClusterPenalizer.

        Args:
            params: The parameters to penalize

            w: The weight to apply to the penalty

            init_ctr: The initial value of c

            description: A string that will be used to identify the penalizer in the string returned
            by __str__()

            learnable_parameters: True if c should be learnable; false if it should be fixed
        """
        super().__init__(params)
        self.w = w
        self.learnable_parameters = learnable_parameters

        if learnable_parameters:
            self.c = torch.nn.Parameter(torch.Tensor([init_ctr]))
        else:
            self.register_buffer('c', torch.Tensor([init_ctr]))
        self.last_p = np.nan
        if description is not None:
            self.description = description
        else:
            self.description = 'Scalar Parameter Penalizer'

    def copy_state_from(self, other):
        """ Copies the state of another penalizer to this penalizer.

        Args:
            other: The other penalizer to copy state form.
        """
        with torch.no_grad():
            self.c.data = other.c.data
        self.last_p = other.last_p

    def clone(self, clean:bool = True):
        """ Returns a copy of self.

        Args:
            clean: If true, attribute values that we might not want to transfer to a new object (such as record of last
            penalty value) will not be copied.

        Returns:
            obj: The new object

        """

        self_copy = copy.deepcopy(self)
        if clean:
            self_copy.last_p = np.nan

        return self_copy

    def check_point(self) -> dict:
        """ Returns a check point dictionary for the penalizer.

        Returns:
            d: A dictionary with the following keys:
                c: The center of the parameter
                last_p: The value of the last penalty that was computed
        """

        c = copy.deepcopy(self.c)
        c = c.detach().cpu().numpy()

        return {'c': c, 'last_p': self.last_p}

    def get_marked_params(self, key: str) -> List[torch.nn.Parameter]:
        """ Returns marked parameters.

        The only parameter of the penalizer is the centers tensor, c, which is marked with the tag 'fast'

        Returns:
            params: A list.  If the key was fast this will hold the 'c' parameter.  If not, this list will be empty.
        """

        if key == 'fast' and self.learnable_parameters:
            return [self.c]
        else:
            return []

    def list_param_keys(self)  -> List[str]:
        """ Returns the list of keys associated with internal, learnable parameters. """
        return ['fast']

    def penalize_and_backwards(self, call_backwards: bool = True) -> torch.Tensor:
        """ Computes the penalty over the parameters and then calls backwards.

        Args:
            call_backwards: True if backwards should be called.

        Returns:
            penalty: The calculated penalty
        """

        penalty = 0.0
        if self.w > 0:
            for p in self.params:
                self.to(p.device)
                cur_penalty = self.w*torch.sum(torch.sum((p - self.c)**2))
                if call_backwards:
                    cur_penalty.backward()
                penalty += cur_penalty.detach().cpu().numpy()

        self.last_p = penalty
        return penalty

    def __str__(self):
        """ Returns a string with the state of the penalizer. """
        return (self.description + ' state'+
                '\n Center: ' + str(self.c.detach().cpu().numpy()) +
                '\n Last Penalty: ' + str(self.last_p))


class UnsignedScalarPenalizer(ScalarPenalizer):
    """ Penalizes the elements of parameters, with the squared distance of each element from a center.

        In particular, given a set of paremters p_0, ..., p_N of arbitrary shape, the penalty computed by this object is:

        w\sum_{i=1}^N ||abs(p_i) - abs(c)||_2^2 where c the scalar center.

    There is only one parameter for this penalizer, which is tagged with 'fast', to indicate that in models trained
    with slow and fast learning rates, we would expect the center to be updated with the fast learning rate.

    """

    def __init__(self, params: Sequence[torch.nn.Parameter], w: float, init_ctr: float,
                 description:str = None, learnable_parameters: bool = True):
        """ Creates a new instance of an UnsignedScalarPenalizer object.

        See __init__() of ScalarPenalizer for documentation of input arguments.
        """

        super().__init__(params=params, w=w, init_ctr=init_ctr, description=description,
                         learnable_parameters=learnable_parameters)

    def penalize_and_backwards(self, call_backwards: bool = True) -> torch.Tensor:
        """ Computes the penalty over the parameters and then calls backwards.

        Args:
            call_backwards: True if backwards should be called.

        Returns:
            penalty: The calculated penalty
        """

        penalty = 0.0
        if self.w > 0:
            for p in self.params:
                self.to(p.device)
                cur_penalty = self.w*torch.sum(torch.sum((torch.abs(p) - torch.abs(self.c))**2))
                if call_backwards:
                    cur_penalty.backward()
                penalty += cur_penalty.detach().cpu().numpy()

        self.last_p = penalty
        return penalty

