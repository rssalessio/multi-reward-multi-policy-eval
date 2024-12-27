from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence



class BaseNetwork(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device
    
    def clone(self, source: BaseNetwork) -> BaseNetwork:
        """ Deecopy the network """
        self.hard_update(source)
        return self
    
    def hard_update(self, source: BaseNetwork) -> None:
        """ Hard update of the network from a source network """
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: BaseNetwork, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        return self.polyak_update(source, tau)

    def polyak_update(self, source: BaseNetwork, tau: float) -> None:
        """ Polyak update (soft update) of the network using a time constant tau """
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def freeze(self) -> BaseNetwork:
        """ Freeze the parameters of the network """
        super().eval()
        # Freeze the target network
        for param in self.parameters():
            param.requires_grad = False
        return self

class SequentialBaseNetwork(BaseNetwork):
    def __init__(self, layers: Sequence[nn.Module], device: torch.device):
        super().__init__(device)
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(x)