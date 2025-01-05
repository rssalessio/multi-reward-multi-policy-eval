import numpy as np
from typing import Any, Optional, Sequence
from numpy.typing import NDArray

class TrajectoryReplayBuffer:
    _data: Optional[Sequence[NDArray]]
    _capacity: int
    _num_added: int
    _trajectory_length: int

    def __init__(self, capacity: int, trajectory_length: int):
        self._data = None
        self._capacity = capacity
        self._num_added = 0
        self._trajectory_length = trajectory_length

    def add(self, trajectory: Sequence[Any]):
        if self._data is None:
            self._preallocate(trajectory)
        
        if len(trajectory[0]) != self._trajectory_length:
            raise ValueError(f"Trajectory length must be {self._trajectory_length}")

        idx = self._num_added % self._capacity
        for slot, traj in zip(self._data, trajectory):
            slot[idx] = np.array(traj)
        
        self._num_added += 1

    def sample(self, batch_size: int) -> Sequence[NDArray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        indices = np.random.choice(self.size, batch_size, replace=False)
        return [slot[indices] for slot in self._data]

    def reset(self):
        self._data = None
        self._num_added = 0

    @property
    def size(self) -> int:
        return min(self._capacity, self._num_added)

    def _preallocate(self, trajectory: Sequence[Any]):
        as_array = [np.asarray(traj) for traj in trajectory]
        self._data = [np.zeros((self._capacity,) + (self._trajectory_length,) + x.shape[1:], dtype=x.dtype) 
                      for x in as_array]
