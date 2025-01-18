from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from multireward_ope.continuous.networks.ensemble_linear_layer import EnsembleLinear
from typing import NamedTuple, Sequence, Optional, Tuple

NumpyObservation = NDArray[np.float64]
NumpyAction = NDArray[np.int64]
NumpyRewards = NDArray[np.float64]

class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NumpyObservation
    a_tm1: int
    a_pi_tm1: int
    a_pi_t: int
    r_t: NumpyRewards
    d_t: float
    o_t: NumpyObservation
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]

class TimeStep(NamedTuple):
    observation: NumpyObservation
    action: int | float
    rewards: NumpyRewards
    done: bool | float
    next_observation: NumpyObservation
    eval_policy_action: int | float
    eval_policy_next_action: int | float
    start: bool

    def to_float32(self) -> TimeStep:
        return TimeStep(
            np.float32(self.observation),
            np.float32(self.action),
            np.float32(self.rewards),
            np.float32(self.done),
            np.float32(self.next_observation),
            np.float32(self.eval_policy_action),
            np.float32(self.eval_policy_next_action),
            np.float32(self.start)
        )

class TorchTransitions(NamedTuple):
    o_tm1: torch.Tensor
    a_tm1: torch.Tensor
    a_pi_tm1: int
    a_pi_t: int
    r_t: torch.Tensor
    d_t: torch.Tensor
    o_t: torch.Tensor
    m_t: Optional[torch.Tensor] = None
    z_t: Optional[torch.Tensor] = None

    @staticmethod
    def from_minibatch(batch: Sequence[NDArray], device: torch.device, num_rewards: int) -> TorchTransitions:
        if len(batch) == 5:
            o_tm1, a_tm1, a_pi_tm1, a_pi_t, r_t, d_t, o_t = batch
            m_t, z_t = None, None
        else:
            o_tm1, a_tm1, a_pi_tm1, a_pi_t, r_t, d_t, o_t, m_t, z_t = batch
            m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False, device=device)
            z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False, device=device)


        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False, device=device)
        a_pi_tm1 = torch.tensor(a_pi_tm1, dtype=torch.int64, requires_grad=False, device=device)
        a_pi_t = torch.tensor(a_pi_t, dtype=torch.int64, requires_grad=False, device=device)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False, device=device)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False, device=device)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False, device=device)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False, device=device)
        

        return TorchTransitions(o_tm1, a_tm1, a_pi_tm1, a_pi_t, r_t, d_t, o_t, m_t, z_t)
    
    def expand_batch(self, ensemble_size: int, num_rewards: int) -> TorchTransitions:
        r_t = self.r_t.unsqueeze(1)
        d_t = self.d_t[..., None, None]
        z_t = self.z_t#.unsqueeze(-1)
        m_t = self.m_t#.unsqueeze(-1)

        a_tm1 = self.a_tm1[..., None, None, None].expand(
                -1, ensemble_size, num_rewards, 1)
        a_pi_tm1 = self.a_pi_tm1[..., None, None, None].expand(
                -1, ensemble_size, num_rewards, 1)
        a_pi_t = self.a_pi_t[..., None, None, None].expand(
                -1, ensemble_size, num_rewards, 1)
        
        return TorchTransitions(self.o_tm1, a_tm1, a_pi_tm1, a_pi_t, r_t, d_t, self.o_t, m_t, z_t)

def expand_obs_with_rewards(obs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    assert rewards.shape[0] == rewards.shape[1], 'Rewards must be a square matrix'
    n,d = obs.shape
    m = rewards.shape[0]
    #[m,n,d]
    obs_exp = obs.unsqueeze(0).expand(m,-1,-1)
    #[m,n,m]
    rew_exp = rewards.unsqueeze(1).expand(-1,n, -1)
    return torch.cat((obs_exp, rew_exp), dim=2).reshape(n * m, m+d)

    
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, EnsembleLinear):
                stddev = np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1]))
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-3*stddev, b=3*stddev)
                torch.nn.init.zeros_(m.bias.data)
