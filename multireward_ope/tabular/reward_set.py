import cvxpy as cp
import numpy as np
import numpy as np
import numpy.typing as npt
from typing import Sequence
from abc import abstractmethod
from enum import Enum

class RewardSetType(object):
    FINITE = 'finite'
    BOX = 'Box'
    CIRCLE = 'Circle'
    POLYTOPE = 'Polytope'
    REWARD_FREE = 'RewardFree'
    NONE =  'None'


class RewardSet(object):
    set_type: RewardSetType
    num_states: int

    def __init__(self, num_states: int):
        self.set_type = RewardSetType.NONE
        self.num_states = num_states

    @abstractmethod
    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        """Check if a reward vecotr satisfies the constriants

        Args:
            reward (npt.NDArray[np.float64]): reward vecotr

        Returns:
            bool: True if the constraints are satisfied
        """
        raise Exception('Not implemented')
    
    @abstractmethod
    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        """Get CVXPY constraints 

        Args:
            var (cp.Variable): Variable to which the constraints are applied

        Returns:
            Sequence[cp.Constraint]: List of constraints
        """
        raise Exception('Not implemented')

class RewardSetCircle(RewardSet):
    """ Lp circle, defined by a tuple (center, radius, p-norm)"""
    center: npt.NDArray[np.float64]
    radius: float
    p: int | str

    def __init__(self, num_states: int, center: npt.NDArray[np.float64], radius: float, p: int | str = 2):
        super().__init__(num_states)
        self.set_type = RewardSetType.CIRCLE
        self.center = center
        self.radius = radius
        self.p = p

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        norm = np.linalg.norm(reward - self.center, ord=self.p)
        return np.all([reward >= 0, reward <= 1, norm <= self.radius])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, cp.norm(var - self.center, self.p) <= self.radius, var <= 1]
        return constraints



class RewardSetPolytope(RewardSet):
    """ We use the H-representation to represent a polytope, Ax<=b"""
    A: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]

    def __init__(self, num_states: int, A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
        super().__init__(num_states)
        self.set_type = RewardSetType.POLYTOPE
        self.A = A
        self.b = b

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        return np.all([reward >= 0, reward <= 1, self.A @ reward <= self.b])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, self.A @ var <= self.b, var <= 1]
        return constraints


class RewardSetBox(RewardSet):
    """ Bounds the reward set as a<=r<=b elementwise """
    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]

    def __init__(self, num_states: int, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
        super().__init__(num_states)
        self.set_type = RewardSetType.BOX
        self.a = a
        self.b = b

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        return np.all([reward >= np.maximum(0, self.a), reward <= np.minimum(1, self.b)])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= np.maximum(0, self.a), var <= np.minimum(1, self.b)]
        return constraints


class RewardSetRewardFree(RewardSet):
    """ Consider the entire set [0,1]^SA """

    def __init__(self, num_states: int):
        super().__init__(num_states)
        self.set_type = RewardSetType.REWARD_FREE

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        return np.all([reward >= 0, reward <= 1])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, var <= 1]
        return constraints

class RewardSetFinite(RewardSet):
    """ Consider a finite set of M rewards, each of size S """

    def __init__(self, num_states: int, rewards: npt.NDArray[np.float64]):
        super().__init__(num_states)
        self.set_type = RewardSetType.FINITE
        self.rewards = rewards
        self.num_rewards = rewards.shape[0]

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        if np.any([reward < 0, reward > 1]): return False

        for i in range(self.num_rewards):
            if np.isclose(np.linalg.norm(reward - self.rewards[i]), 0):
                return True
        return False

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, var <= 1]
        return constraints
