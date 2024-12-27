import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from multireward_ope.continuous.agents.utils.base_networks import BaseNetwork
from agents.utils import weight_init
from multireward_ope.continuous.agents.utils.ensemble_linear_layer import EnsembleLinear
from typing import NamedTuple
from functools import partial


class Values(NamedTuple):
    """ Contains the Q-values and the M-values"""
    q_values: torch.Tensor
    m_values: torch.Tensor

def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int, final_activation = nn.ReLU) -> nn.Module:
    """ Create a single network """
    net = [
        EnsembleLinear(input_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, output_size)]
    if final_activation is not None:
        net.append(final_activation())
    
    return nn.Sequential(*net)


class EnsembleSingleNetwork(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, hidden_size: int, ensemble_size: int, final_activation: nn.Module, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
        self.num_rewards = num_rewards
        self.ensemble_size = ensemble_size
        self.num_actions = num_actions
        self._trunk =  make_single_network(state_dim, num_actions * num_rewards, hidden_size, ensemble_size, final_activation=final_activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward a batch of states

        Args:
            x (torch.Tensor): batch of states of dim (s,n,d), where
                            - s is the ensemble size
                            - n is the batch size
                            - d is the number of features

        Returns:
            torch.Tensor: returns a tensor of dim [s, n, r, a], where 
                            - s is the ensemble size
                            - n is the batch size
                            - r is the number of rewards
                            - a is the number of actions

        """
        return self._trunk(x).view(self.ensemble_size, -1, self.num_rewards, self.num_actions)

class EnsembleWithPrior(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, prior_scale: float,
                 ensemble_size: int, hidden_size: int, final_activation: nn.Module, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
        self.ensemble_size = ensemble_size
        self._network = EnsembleSingleNetwork(state_dim, num_rewards, num_actions, hidden_size, ensemble_size, final_activation, device, generator)
        self._prior_network = EnsembleSingleNetwork(state_dim, num_rewards, num_actions, hidden_size, ensemble_size, final_activation, device, generator)
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


class ValueEnsembleWithPrior(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, prior_scale: float, ensemble_size: int, hidden_size: int, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
        self.num_rewards = num_rewards
        self.ensemble_size = ensemble_size
        self._q_network = EnsembleWithPrior(state_dim, num_rewards, num_actions, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=None, device=device, generator=generator)
        self._m_network = EnsembleWithPrior(state_dim, num_rewards, num_actions, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=nn.ReLU, device=device, generator=generator)
        
        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev, generator=generator)
                torch.nn.init.zeros_(m.bias.data)
        
        self.to(device).apply(init_weights)

    
    def forward(self, x: torch.Tensor) -> Values:
        q = self._q_network.forward(x)
        m = self._m_network.forward(x)
        return Values(q, m)