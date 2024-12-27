# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, NamedTuple, Optional, Dict
from dataclasses import dataclass

from agents.utils import TimeStep

@dataclass
class DeepSeaConfig(object):
    size: int = 10
    slipping_probability: float = 0.05
    randomize: bool = True

    def build(self) -> DeepSea:
        return DeepSea(self)
    
    @staticmethod
    def name(cls: DeepSeaConfig) -> str:
        return f'{cls.size}'



class DeepSea(object):
    NAME = 'DeepSea'
    def __init__(self, cfg: DeepSeaConfig):
        self._cfg = cfg
        self._size = cfg.size
        self._slipping_probability = cfg.slipping_probability
        self.base_rewards = np.zeros((cfg.size, cfg.size, cfg.size))
        self.base_rewards[:,-1,:] = np.eye(cfg.size)

        self.random_rewards = None
        self.rewards = self.base_rewards
        self.num_base_rewards = cfg.size
        self.num_rewards = cfg.size

        self._column = 0
        self._row = 0

        self.visits = np.zeros((cfg.size, cfg.size))
        

        if cfg.randomize:
            self._action_mapping = np.random.binomial(1, 0.5, cfg.size)
        else:
            self._action_mapping = np.ones(cfg.size)

        self._done = False


    def compute_Q_values(self, rewards: NDArray[np.float32], gamma: float):
        Q = np.zeros((rewards.shape[0], self._size, self._size, 2))
        p = self._slipping_probability

        for i in range(rewards.shape[0]):
            for row in reversed(range(self._size - 1)):
                for column in range(self._size):
                    if column >= row + 1: break
                    next_row = row + 1
                    
                    for action in [0,1]:
                        if action:  # right
                            next_column = np.clip(column + 1, 0, self._size - 1)
                            inv_col= np.clip(column - 1, 0, self._size - 1)
                        else:  # left
                            next_column= np.clip(column - 1, 0, self._size - 1)
                            inv_col= np.clip(column + 1, 0, self._size - 1)
                        Q[i, row, column, action] = (1-p) * (rewards[i, next_row, next_column] + gamma * Q[i, next_row, next_column].max()) + p * (rewards[i, next_row, inv_col] + gamma * Q[i, next_row, inv_col].max())
        return Q

    def step(self, action: int) -> Tuple[TimeStep, Dict]:
        _current_observation = self._get_observation(self._row, self._column)
        
        if np.random.uniform() < self._slipping_probability:
            action = int(not action)

        if self._done:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(_current_observation, action, 0, True, observation)

        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]


        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)

        
        # Compute the observation
        self._row += 1
        done = self._row == self._size -1 

        rewards = self.compute_rewards()
        self.visits[self._row, self._column] += 1
        observation = self._get_observation(self._row, self._column)
        return TimeStep(_current_observation, action, rewards, done, observation), {}

    def reset(self) -> NDArray[np.float32]:
        self._done = False
        self._column = 0
        self._row = 0
        observation = self._get_observation(self._row, self._column)
        self.visits[self._row, self._column] += 1
        
        return observation

    def _get_observation(self, row: int, column: int) -> NDArray[np.float32]:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1

        return observation.flatten()

    @property
    def obs_shape(self) -> Tuple[int, int]:
        return self._size, self._size

    @property
    def num_actions(self) -> int:
        return 2

    def compute_rewards(self) -> NDArray[np.float64]:
        return self.rewards[:, self._row, self._column]


