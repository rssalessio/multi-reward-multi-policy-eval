from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from multireward_ope.tabular.mdp import MDP
from typing import Tuple, NamedTuple

@dataclass
class RiverSwimParameters(object):
    num_states: int = 5

    def build(self) -> RiverSwim:
        return RiverSwim(self)

# class DoubleChainParameters(NamedTuple):
#     length:int = 5
#     p: float = 0.7

# class NArmsParameters(NamedTuple):
#     num_arms: int = 6
#     p0: float = 1.0

# class ForkedRiverSwimParameters(NamedTuple):
#     river_length: int = 5



class RiverSwim(MDP):
    """RiverSwim environment
    @See also https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1374179
    """
    current_state: int                      # Current state
    
    def __init__(self, 
                 parameters: RiverSwimParameters):
        """Initialize a river swim environment

        Parameters
        ----------
        parameters : RiverSwimParameters
            Parameters of the environment
        """        
        ns = parameters.num_states
        na = 2
    
        transitions = np.zeros((ns, na, ns))
        
        # Create transitions
        for s in range(1, ns-1):
            transitions[s, 1, s] = 0.6
            transitions[s, 1, s-1] = 0.1
            transitions[s, 1, s+1] = 0.3
        transitions[1:-1, 0, 0:-2] = np.eye(ns-2)

        transitions[0, 0, 0] = 1
        transitions[0, 1, 0] = 0.7
        transitions[0, 1, 1] = 0.3
        transitions[-1, 1, -1] = 0.3
        transitions[-1, 1, -2] = 0.7
        transitions[-1, 0, -2] = 1
        
        super().__init__(transitions)
        # Reset environment
        self.reset()

    def build_reward_matrix(self, min_reward: float = 0.05, max_reward: float = 1.0):
        rewards = np.zeros((self.dim_state, self.dim_action))
        
        # Create rewards
        rewards[0, 0] = min_reward
        rewards[-1, 1] = max_reward
        return rewards
    
    def reset(self) -> int:
        """Reset the current state

        Returns
        -------
        int
            initial state 0
        """        
        self.current_state = 0
        return self.current_state
    
    def step(self, action: int, reward: npt.NDArray[np.float64] | None = None) -> Tuple[int, float]:
        """Take a step in the environment

        Parameters
        ----------
        action : int
            An action (0 or 1)

        Returns
        -------
        Tuple[int, float]
            Next state and reward
        """        
        assert action == 0 or action == 1, 'Action needs to either 0 or 1'
        
        next_state = np.random.choice(self.dim_state, p=self.P[self.current_state, action])
        rew = None if reward is None else reward[self.current_state, action]
        self.current_state = next_state
        return next_state, rew
    