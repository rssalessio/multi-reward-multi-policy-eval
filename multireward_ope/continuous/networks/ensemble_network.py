import torch
import torch.nn as nn
from multireward_ope.continuous.networks.base_networks import BaseNetwork
from multireward_ope.continuous.networks.networks_utils import make_single_network



class EnsembleSingleNetwork(BaseNetwork):
    def __init__(self, state_dim: int, num_rewards: int, num_actions: int, hidden_size: int, ensemble_size: int, final_activation: nn.Module, device: torch.device):
        super().__init__(device)
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