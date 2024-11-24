from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Sequence

Policy = npt.NDArray[np.ulong]

class PolicyFactory:
    @staticmethod
    def from_sequence(x: Sequence) -> Policy:
        return np.array([np.astype(z, np.ulong) for z in x], dtype=np.ulong)
    
    @staticmethod
    def random(dim_state: int, dim_action: int) -> Policy:
        return np.random.choice(dim_action, size=dim_state).astype(dtype=np.ulong)
    

