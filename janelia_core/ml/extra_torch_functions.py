""" Contains custom torch functions. """

import torch

# Define constants
LOG_2 = torch.log(torch.tensor(2.0))


def log_cosh(x: torch.Tensor) -> torch.Tensor:
    """ Computes log cosh(x) in a numerically stable way.

    Args:
        x: Input values

    Returns:
        y: Output values
    """

    abs_x = torch.abs(x)
    return torch.abs(x) - LOG_2 + torch.log1p(torch.exp(-2*abs_x))

