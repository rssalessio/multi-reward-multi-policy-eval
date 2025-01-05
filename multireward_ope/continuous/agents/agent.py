from __future__ import annotations
import abc
import numpy as np
import torch
from multireward_ope.continuous.agents.utils.utils import NumpyObservation, TimeStep
from multireward_ope.continuous.agents.utils.replay_buffer import ReplayBuffer
from multireward_ope.continuous.agents.utils.per_buffer import PrioritizedReplayBuffer
from multireward_ope.continuous.agents.utils.trajectory_replay_buffer import TrajectoryReplayBuffer

from datetime import datetime

class Agent(abc.ABC):
    """
    Base class of an exploration agent
    """
    NAME = 'AbstractAgent'
    def __init__(self, epsilon: float, capacity: int, horizon: int, device: torch.device):
        self._epsilon = epsilon
        self._replay =  TrajectoryReplayBuffer(capacity=capacity, trajectory_length=horizon)#PrioritizedReplayBuffer(capacity=capacity)#
        
        self._device = device

    def eps_fn(self, steps: int) -> float:
        """ Epsilon-based exploration 
            Computes epsilon =  x0/(x0+t)
        """
        x0 = self._epsilon
        return min(1, x0 / max(1, (steps + x0)))
    
    @property
    def buffer(self) -> ReplayBuffer:
        return self._replay

    @abc.abstractmethod
    def qvalues(self, observation: TimeStep) -> np.ndarray:
        pass

    @abc.abstractmethod
    def select_action(self, observation: TimeStep, step: int) -> int:
        pass
    
    @abc.abstractmethod
    def select_greedy_action(self, observation: TimeStep) -> int:
        pass

    @abc.abstractmethod
    def update(self, timestep: TimeStep) -> None:
        pass
    
    @abc.abstractstaticmethod
    def make_default_agent(
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: any,
            device: torch.device) -> Agent:
        raise NotImplementedError('make_default_agent not implemented')


    @abc.abstractmethod
    def save_model(self, path: str, seed: int, step: int):
        raise Exception('Not implemented')