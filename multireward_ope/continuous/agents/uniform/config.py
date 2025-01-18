from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class UniformConfig:
    hidden_layer_size: int = 32
    batch_size: int = 128
    discount: float = 0.99
    replay_capacity: int = 100000
    min_replay_size: int = 128
    sgd_period: int = 1
    target_soft_update: float = 1e-2
    lr_Q: float = 1e-4
    epsilon_0: float = 10
    exploration_prob: float = 0.2

    @staticmethod
    def name(cls: UniformConfig) -> str:
        return ''