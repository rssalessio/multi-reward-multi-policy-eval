import cvxpy as cp
import numpy as np
import numpy as np
import numpy.typing as npt
from typing import Sequence
from abc import abstractmethod
from enum import Enum
from typing import NamedTuple

class RewardSetType(Enum):
    FINITE = 'finite'
    BOX = 'Box'
    CIRCLE = 'Circle'
    POLYTOPE = 'Polytope'
    REWARD_FREE = 'RewardFree'
    NONE =  'None'

class RewardSet(object):
    set_type: RewardSetType
    num_states: int
    num_actions: int

    def __init__(self, num_states: int, num_actions: int, set_type: RewardSetType):
        self.config = set_type
        self.set_type = set_type
        self.num_states = num_states
        self.num_actions = num_actions

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
    
    def sample(self, n: int) -> Sequence[npt.NDArray[np.float64]]:
        """Sample n reward vectors from the set

        Args:
            n (int): number of rewards

        Returns:
            Sequence[npt.NDArray[np.float64]]: Sequence of reward vectors of size S
        """
        D = self.num_states
        rewards = []
        for i in range(n):
            while True:
                r = np.random.uniform(size=D)
                if self.satisfy_constraints(r):
                    break
            rewards.append(r)
        return rewards
    
    def project_reward(self, r: npt.NDArray[np.float64], solver: str = cp.CLARABEL) -> npt.NDArray[np.float64]:
        """Project a reward onto the set of admissible rewards in the l2 norm sense

        Args:
            r (npt.NDArray[np.float64]): reward to project
            solver (str, optional): solver for CVXPY. Defaults to cp.CLARABEL.

        Returns:
            npt.NDArray[np.float64]: projected reward vector
        """
        proj_r = cp.Variable(self.num_states, nonneg=True)

        obj = cp.norm(r - proj_r)
        constraints = self.get_constraints(proj_r)
        problem = cp.Problem(cp.Minimize(obj), constraints)
        res = problem.solve(solver=solver)
        return proj_r.value


class RewardSetCircle(RewardSet):
    """ Lp circle, defined by a tuple (center, radius, p-norm)"""

    class RewardSetCircleConfig(NamedTuple):
        center: npt.NDArray[np.float64]
        radius: float
        p: int | str
    
    config: RewardSetCircleConfig

    def __init__(self, num_states: int, num_actions: int, config: RewardSetCircleConfig):
        super().__init__(num_states, num_actions, RewardSetType.CIRCLE)
        self.config = config

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        norm = np.linalg.norm(reward - self.config.center, ord=self.config.p)
        constraints = [reward >= 0, reward <= 1, norm <= self.config.radius]
        return np.all([np.all(c) for c in constraints])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, cp.norm(var - self.config.center, self.config.p) <= self.config.radius, var <= 1]
        return constraints



class RewardSetPolytope(RewardSet):
    """ We use the H-representation to represent a polytope, Ax<=b"""
    
    class RewardSetPolytopeConfig(NamedTuple):
        A: npt.NDArray[np.float64]
        b: npt.NDArray[np.float64]
    
    config: RewardSetPolytopeConfig

    def __init__(self, num_states: int, num_actions: int, config: RewardSetPolytopeConfig):
        super().__init__(num_states, num_actions, RewardSetType.POLYTOPE)
        self.config = config

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        constraints = [reward >= 0, reward <= 1, self.config.A @ reward <= self.config.b]
        return np.all([np.all(c) for c in constraints])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, self.config.A @ var <= self.config.b, var <= 1]
        return constraints


class RewardSetBox(RewardSet):
    """ Bounds the reward set as a<=r<=b elementwise """
    
    class RewardSetBoxConfig(NamedTuple):
        a: npt.NDArray[np.float64]
        b: npt.NDArray[np.float64]
    
    config: RewardSetBoxConfig

    def __init__(self, num_states: int, num_actions: int, config: RewardSetBoxConfig):
        super().__init__(num_states, num_actions, RewardSetType.BOX)
        self.config = config

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        constraints = [reward >= np.maximum(0, self.config.a), reward <= np.minimum(1, self.config.b)]
        return np.all([np.all(c) for c in constraints])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= np.maximum(0, self.config.a), var <= np.minimum(1, self.config.b)]
        return constraints


class RewardSetRewardFree(RewardSet):
    """ Consider the entire set [0,1]^SA """

    class RewardSetFreeConfig(NamedTuple):
        pass

    def __init__(self, num_states: int, num_actions: int, config: RewardSetFreeConfig):
        super().__init__(num_states, num_actions, RewardSetType.REWARD_FREE)
        self.config = config

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        constraints = [reward >= 0, reward <= 1]
        return np.all([np.all(c) for c in constraints])

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, var <= 1]
        return constraints

class RewardSetFinite(RewardSet):
    """ Consider a finite set of M rewards, each of size S """

    class RewardSetFiniteConfig(NamedTuple):
        rewards: npt.NDArray[np.float64]
    
    config: RewardSetFiniteConfig

    def __init__(self, num_states: int, num_actions: int, config: RewardSetFiniteConfig):
        super().__init__(num_states, num_actions, RewardSetType.FINITE)
        self.num_rewards = self.config.rewards.shape[0]
        self.config = config

    def satisfy_constraints(self, reward: npt.NDArray[np.float64]) -> bool:
        if np.any([np.any(reward < 0), np.any(reward > 1)]): return False

        for i in range(self.num_rewards):
            if np.isclose(np.linalg.norm(reward - self.config.rewards[i]), 0):
                return True
        return False

    def get_constraints(self, var: cp.Variable) -> Sequence[cp.Constraint]:
        constraints = [var >= 0, var <= 1]
        return constraints



RewardSetConfig = RewardSetBox.RewardSetBoxConfig | \
                  RewardSetCircle.RewardSetCircleConfig | \
                  RewardSetRewardFree.RewardSetFreeConfig | \
                  RewardSetPolytope.RewardSetPolytopeConfig | \
                  RewardSetFinite.RewardSetFiniteConfig
