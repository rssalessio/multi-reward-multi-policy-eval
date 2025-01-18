import torch
import torch.nn as nn
from multireward_ope.continuous.networks.base_networks import BaseNetwork
from multireward_ope.continuous.agents.utils.utils import weight_init
from typing import NamedTuple
from multireward_ope.continuous.networks.ensemble_with_prior import EnsembleWithPrior


class Values(NamedTuple):
    """ Contains the Q-values and the M-values"""
    q_values: torch.Tensor
    m_values: torch.Tensor

class ValueEnsembleWithPrior(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, prior_scale: float, ensemble_size: int, hidden_size: int, device: torch.device):
        super().__init__(device)
        self.num_rewards = num_rewards
        self.ensemble_size = ensemble_size
        self.num_actions = num_actions
        self._q_network = EnsembleWithPrior(state_dim, num_rewards, num_actions, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=None, device=device)
        self._m_network = EnsembleWithPrior(state_dim, num_rewards, num_actions, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=None, device=device)
        self._m_activation = nn.Softplus()
        self.to(device).apply(weight_init)

    
    def forward(self, x: torch.Tensor) -> Values:
        q = self._q_network.forward(x)
        m =self._m_network.forward(x)
        return Values(q, m)