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
        return np.any([reward < 0, norm > self.radius])

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
        return np.any([reward < 0, self.A @ reward > self.b])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, self.A @ var <= self.b, var <= 1]
        return constraints

