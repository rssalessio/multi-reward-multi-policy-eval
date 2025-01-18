import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from multireward_ope.continuous.networks.base_networks import BaseNetwork
from multireward_ope.continuous.agents.utils.utils import weight_init
from multireward_ope.continuous.networks.ensemble_linear_layer import EnsembleLinear
from typing import NamedTuple
from functools import partial
from multireward_ope.continuous.networks.ensemble_network import EnsembleSingleNetwork



class EnsembleWithPrior(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, prior_scale: float,
                 ensemble_size: int, hidden_size: int, final_activation: nn.Module, device: torch.device):
        super().__init__(device)
        self.ensemble_size = ensemble_size
        self._network = EnsembleSingleNetwork(state_dim, num_rewards, num_actions, hidden_size, ensemble_size, final_activation, device)
        self._prior_network = EnsembleSingleNetwork(state_dim, num_rewards, num_actions, hidden_size, ensemble_size, final_activation, device)
        self._prior_scale = prior_scale

        # Freeze training of the prior network
        self._prior_network.freeze()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward a batch of states

        Args:
            x (torch.Tensor): batch of states of dim (n,d), where
                            - n is the batch size
                            - d is the number of features

        Returns:
            torch.Tensor: returns a tensor of dim [n, s, r, a], where
                            - n is the batch size
                            - s is the ensemble size
                            - r is the number of rewards
                            - a is the number of actions

        """
        # Prepare tensor for the ensemble
        x = x[None, ...].expand(self.ensemble_size, -1, -1)
        # Swap axes so first dim is the batch size
        values = self._network.forward(x).swapaxes(0,1)
        prior_values = self._prior_network.forward(x).swapaxes(0,1)
        return values + self._prior_scale * prior_values.detach()