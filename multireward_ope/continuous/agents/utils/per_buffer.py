from __future__ import annotations
import numpy as np
from typing import Any, Optional, Sequence
from numpy.typing import NDArray


class PrioritizedReplayBuffer:
    _data: Optional[Sequence[NDArray]]
    _capacity: int
    _num_added: int
    _priorities: NDArray
    _alpha: float

    def __init__(self, capacity: int, alpha: float = 0.6):
        self._data = None
        self._capacity = capacity
        self._num_added = 0
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._alpha = alpha

    def add(self, items: Sequence[Any], priority: float = 1.0):
        if self._data is None:
            self._preallocate(items)

        idx = self._num_added % self._capacity
        for slot, item in zip(self._data, items):
            slot[idx] = item

        max_priority = self._priorities.max() if self._num_added > 0 else priority
        self._priorities[idx] = max_priority

        self._num_added += 1

    def sample(self, size: int, beta: float = 0.4) -> tuple[Sequence[NDArray], NDArray, NDArray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        scaled_priorities = self._priorities[:self.size] ** self._alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(self.size, size=size, p=probs)
        importance_weights = (1 / (self.size * probs[indices])) ** beta
        importance_weights /= importance_weights.max()

        sampled_data = [slot[indices] for slot in self._data]
        return sampled_data, indices, importance_weights

    def update_priorities(self, indices: NDArray, priorities: NDArray):
        self._priorities[indices] = np.maximum(priorities, 1e-6)

    def reset(self):
        self._data = None
        self._priorities.fill(0)
        self._num_added = 0

    @property
    def size(self) -> int:
        return min(self._capacity, self._num_added)

    def _preallocate(self, items: Sequence[Any]):
        as_array = [np.asarray(item) for item in items]
        self._data = [np.zeros((self._capacity,) + x.shape, dtype=x.dtype) for x in as_array]
