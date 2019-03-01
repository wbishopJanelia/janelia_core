""" Contains basic torch modules, supplementing those natviely in Torch.
"""

import torch


class Bias(torch.nn.ModuleList):
    """ Applies a bias transformation to the data y = x + o """

    def __init__(self, d: int):
        """ Creates a Bias object.

        Args:
            d: The dimensionality of the input and output
        """

        super().__init__()
        o = torch.nn.Parameter(torch.zeros(d), requires_grad=True)
        torch.nn.init.normal_(o, std=.01)
        self.register_parameter('o', o)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """ Computes output given input.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor
        """
        return x + self.o
