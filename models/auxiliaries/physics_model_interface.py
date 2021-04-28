from abc import abstractclassmethod
import torch
import torch.nn as nn

class PhysicsModel(nn.Model):
    def __init__(self):
        super(PhysicsModel, self).__init__()

    @abstractclassmethod
    def forward(parameters: torch.Tensor):
        """
        Builds the output using the given parameters.

        Parameters:
        -----------
            - parameters (torch.Tensor): NxM tensor of M parameters for N samples in the range between 0 and 1.

        Returns:
        --------
            - N samples
        """
        raise NotImplementedError()

    @abstractclassmethod
    def get_num_out_channels() -> int:
        """
        Returns the number of parameters that need to be predicted in order to be used by the physics model
        """
        raise NotImplementedError()

    @abstractclassmethod
    def quantity_to_param(self, quantities: torch.Tensor) -> torch.Tensor:
        """
        Transforms the given quantities to parameters by normalizing them to the range [0,1]

        Parameters:
        -----------
            - quantities (torch.Tensor): NxM tensor of M quantities for N samples.

        Returns:
        --------
            - NxM tensor containing the parameter values for the given quantities
        """
        raise NotImplementedError()

    @abstractclassmethod
    def param_to_quantity(self, params: torch.Tensor) -> torch.Tensor:
        """
        Transforms the given parameters to quantities by projecting them into the correct range

        Parameters:
        -----------
            - parameters (torch.Tensor): NxM tensor of M quantities for N samples in the range [0,1].

        Returns:
        --------
            - NxM tensor containing the quantity values for the given parameters
        """
        raise NotImplementedError()
