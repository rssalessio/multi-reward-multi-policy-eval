from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Sequence, Optional, Tuple

NumpyObservation = NDArray[np.float64]
NumpyAction = NDArray[np.int64]
NumpyRewards = NDArray[np.float64]

class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NumpyObservation
    a_tm1: int
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

    def to_float32(self) -> TimeStep:
        return TimeStep(
            np.float32(self.observation),
            np.float32(self.action),
            np.float32(self.rewards),
            np.float32(self.done),
            np.float32(self.next_observation)
        )

class TorchTransitions(NamedTuple):
    o_tm1: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor
    d_t: torch.Tensor
    o_t: torch.Tensor
    m_t: Optional[torch.Tensor] = None
    z_t: Optional[torch.Tensor] = None

    @staticmethod
    def from_minibatch(batch: Sequence[NDArray], device: torch.device, num_rewards: int) -> TorchTransitions:
        if len(batch) == 5:
            o_tm1, a_tm1, r_t, d_t, o_t = batch
            m_t, z_t = None, None
        else:
            o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = batch
            m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False, device=device)
            z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False, device=device)


        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False, device=device)
        r_t = torch.tensor(r_t[:, :num_rewards], dtype=torch.float32, requires_grad=False, device=device)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False, device=device)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False, device=device)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False, device=device)
        

        return TorchTransitions(o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t)
    
    def expand_batch(self, ensemble_size: int, num_rewards: int) -> TorchTransitions:
        r_t = self.r_t.unsqueeze(1)
        d_t = self.d_t[..., None, None]
        z_t = self.z_t.unsqueeze(-1)
        m_t = self.m_t.unsqueeze(-1)

        a_tm1 = self.a_tm1[..., None, None, None].expand(
                -1, ensemble_size, num_rewards, 1)
        
        return TorchTransitions(self.o_tm1, a_tm1, r_t, d_t, self.o_t, m_t, z_t)

def expand_obs_with_rewards(obs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    assert rewards.shape[0] == rewards.shape[1], 'Rewards must be a square matrix'
    n,d = obs.shape
    m = rewards.shape[0]
    #[m,n,d]
    obs_exp = obs.unsqueeze(0).expand(m,-1,-1)
    #[m,n,m]
    rew_exp = rewards.unsqueeze(1).expand(-1,n, -1)
    return torch.cat((obs_exp, rew_exp), dim=2).reshape(n * m, m+d)

    
def weight_init(m, generator: torch.Generator):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, generator=generator)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain, generator=generator)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S



class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms
        self.knn_rms = knn_rms
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.knn_clip = knn_clip
        self.device = device

    def __call__(self, rep):
        source = target = rep
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(self.device)
            ) if self.knn_clip >= 0.0 else reward  # (b1, 1)
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            reward /= self.rms(reward)[0] if self.knn_rms else 1.0
            reward = torch.maximum(
                reward - self.knn_clip,
                torch.zeros_like(reward).to(
                    self.device)) if self.knn_clip >= 0.0 else reward
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1, keepdim=True)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward
    
class RollingMinimum(object):
    def __init__(self, m: int, n: int, min: float):
        self.m = m  # Number of classes
        self.n = n  # Size of each deque
        self.data = np.full((m, n), min, order='C', dtype=np.float64)  # Initialize with infinity
        self.index = 0  # Current indexes for insertion
        self.mins = np.full(m, min, order='C', dtype=np.float64)  # Running minimums
        self._def_min = min
        
    def add(self, points: NDArray[np.float64]):
        self.data[:, self.index % self.n] = np.maximum(self._def_min, points)
        self.index += 1
        self.mins = np.mean(self.data, axis=-1)#np.quantile(self.data, 0.1, axis=-1)
    
    def get_running_mins(self) -> NDArray[np.float64]:
        return self.mins
    
def compute_model_parameters(model) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])