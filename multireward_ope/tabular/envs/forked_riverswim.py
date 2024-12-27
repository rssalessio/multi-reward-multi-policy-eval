from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Tuple
from multireward_ope.tabular.mdp import MDP

@dataclass
class ForkedRiverSwimConfig(object):
    river_length: int = 5

    def build(self) -> ForkedRiverSwim:
        return ForkedRiverSwim(self)
    
    @staticmethod
    def name(cls: ForkedRiverSwimConfig) -> str:
        return f'{cls.river_length}'


class ForkedRiverSwim(MDP):
    """Forked RiverSwim environment
    Like the RiverSwim environment but with 2 rivers
    
    0 1 2 3 4 <- 1st branch
    | | | |
    - 5 6 7 8 <- 2nd branch
    """
    current_state: int                      # CUrrent state
    
    _LEFT: int = 0
    _RIGHT: int = 1
    _SWITCH: int = 2
    
    def __init__(self, cfg: ForkedRiverSwimConfig):
        """Initialize a forked river swim environment

        Parameters
        ----------
        river_length : int, optional
            Length of each river branch
        """        
        
        self.ns = 1 + (cfg.river_length - 1) * 2
        self.na = 3
        
        self._end_river_1 = cfg.river_length - 1
        self._end_river_2 = self.ns - 1
        self._start = 0
        
        transitions = np.zeros((self.ns, self.na, self.ns))
        
        
        # Create transitions
        for start, end in [(1, self._end_river_1), (self._end_river_1 + 1, self._end_river_2)]:
            for s in range(start, end):
                transitions[s, self._RIGHT, s] = 0.6
                transitions[s, self._RIGHT, s-1] = 0.1
                transitions[s, self._RIGHT, s+1] = 0.3
                
                other_side = s + cfg.river_length - 1 if s < self._end_river_1 else s - cfg.river_length + 1
                transitions[s, self._SWITCH, other_side] = 1
            
        transitions[1:self._end_river_1, self._LEFT, 0:self._end_river_1-1] = np.eye(cfg.river_length - 2)
        transitions[self._end_river_1+2:self._end_river_2, self._LEFT, self._end_river_1+1:self._end_river_2-1] = np.eye(cfg.river_length - 3)
        transitions[self._end_river_1+1, self._LEFT, 0] = 1

        transitions[0, self._LEFT, 0] = 1
        transitions[0, self._RIGHT, 0] = 0.7
        transitions[0, self._RIGHT, 1] = 0.3
        for end in [self._end_river_1, self._end_river_2]:
            transitions[end, self._RIGHT, end] = 0.3
            transitions[end, self._RIGHT, end-1] = 0.7
            transitions[end, self._LEFT, end-1] = 1
            
        transitions[self._end_river_1, self._SWITCH, self._end_river_1] = 1
        transitions[self._end_river_2, self._SWITCH, self._end_river_2] = 1
        transitions[self._start, self._SWITCH, self._start] = 1

        super().__init__(transitions)
 
        
        # Reset environment
        self.reset()
    
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
        assert action == 0 or action == 1 or action == 2, 'Action needs to either 0 or 1'
        
        next_state = np.random.choice(self.ns, p=self.P[self.current_state, action])
        reward = None if reward is None else reward[self.current_state, action]
        self.current_state = next_state
        return next_state, reward
    

    def default_policy(self, discount_factor: float) -> np.ndarray:
        R = np.zeros((self.dim_state, self.dim_action))
        R[-1,1]=1
        _,pi,_ = self.policy_iteration(R, discount_factor=discount_factor)
        return pi