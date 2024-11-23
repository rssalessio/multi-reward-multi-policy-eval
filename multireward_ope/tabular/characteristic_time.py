import cvxpy as cp
import numpy as np
import dccp
import numpy.typing as npt
import multiprocessing as mp
from enum import Enum
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.reward_set import RewardSet, RewardSetCircle, RewardSetType, RewardSetRewardFree, RewardSetBox
from typing import NamedTuple



class BoundResult(NamedTuple):
    value: float
    w: npt.NDArray[np.float64]

class CharacteristicTimeSolver(object):
    """ Class to solve the characteristic time problem """
    dim_state: int
    dim_actions: int
    theta: cp.Variable
    KG: cp.Parameter
    e_i: cp.Parameter
    MD_problem: cp.Problem
    rewards: RewardSet
    solver: str
    prev_theta: npt.NDArray[np.float64]
    prev_omega: npt.NDArray[np.float64] | None
    prev_A: npt.NDArray[np.float64]
    MAX_ITER: int = 500

    def __init__(self, dim_state: int, dim_actions: int, solver: str = cp.ECOS):
        self.dim_state = dim_state
        self.dim_actions = dim_actions
        self.theta = cp.Variable(dim_state, nonneg=True)
        self.KG = cp.Parameter((dim_state, dim_state))
        self.e_i = cp.Parameter(dim_state)
        self.solver = solver
        self.prev_theta = np.zeros((dim_state, dim_state))
        self.prev_A = np.zeros((dim_state))

    def build_problem(self, rewards: RewardSet):
        """Build problem to improve speed

        Args:
            rewards (RewardSet): Set of rewards considered
        """
        Ks = self.e_i.T @ self.KG @ self.theta
        obj = cp.Maximize(cp.abs(Ks))
        self.MD_problem = cp.Problem(obj, rewards.get_constraints(self.theta))
        self.rewards = rewards

    def solve(self,
              gamma: float, 
              mdp: MDP,
              policy: npt.NDArray[np.long]) -> BoundResult:
        """Solve the characteristic time optimization problem

        Args:
            gamma (float): discount facotr
            mdp (MDP): MDP considered
            policy (npt.NDArray[np.long]): deterministic policy of size S

        Returns:
            BoundResult: A tuple containing the value of the problem and the optimal solution
        """
        if self.rewards.set_type == RewardSetType.FINITE:
            raise Exception('Finite set not implemented')
        elif self.rewards.set_type == RewardSetType.REWARD_FREE:
            return self._solve_rewardfree(gamma, mdp, policy)
        else:
            return self._solve_general(gamma, mdp, policy)
        

    def evaluate(self,
              omega: npt.NDArray[np.float64],
              gamma: float,
              epsilon: float, 
              mdp: MDP,
              policy: npt.NDArray[np.long]) -> float:
        """Solve the characteristic time optimization problem

        Args:
            omega (npt.NDArray[np.float64]): state-action distribution to evaluate
            gamma (float): discount factor
            epsilon (float): accuracy level
            mdp (MDP): MDP considered
            policy (npt.NDArray[np.long]): deterministic policy of size S

        Returns:
            BoundResult: A tuple containing the value of the problem and the optimal solution
        """
        obj = np.multiply(self.prev_A, omega[np.arange(mdp.dim_state), policy]).max()
        obj *= (gamma / (2 * epsilon * (1 -  gamma))) ** 2
        return obj

        
    def _solve(self, A: npt.NDArray[np.float64], gamma: float, mdp: MDP, policy: npt.NDArray[np.long]):
        normalization = 1 - gamma
        omega = cp.Variable((self.dim_state, self.dim_actions), nonneg=True)
        if self.prev_omega:
            omega.value = self.prev_omega

        constraints = [cp.sum(omega) == 1]
        constraints.extend(
            [cp.sum(omega[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], omega)) for s in range(self.dim_state)])
    
        omega_pi = omega[np.arange(mdp.dim_state), policy]
        obj = cp.multiply(A, cp.inv_pos(omega_pi)) * normalization

        obj = cp.Minimize(cp.max(obj))
        T_problem = cp.Problem(obj, constraints)
        res = T_problem.solve(solver=self.solver)
        self.prev_omega = omega.value
        return BoundResult(res / normalization, omega.value)

        
    def _solve_rewardfree(self,
              gamma: float, 
              mdp: MDP,
              policy: npt.NDArray[np.long]) -> BoundResult:
        G = mdp.build_stationary_matrix(policy, gamma=gamma)
        K = mdp.build_K(policy)

        A = np.zeros((mdp.dim_state, mdp.dim_state))
        for i in range(mdp.dim_state):
            for s in range(mdp.dim_state):
                KG = K[s] @ G
                Ai_s = KG[i]
                pos_idxs = Ai_s >= 0
                
                A[s,i] = np.maximum(Ai_s[pos_idxs].sum(-1), Ai_s[~pos_idxs].sum(-1))
        A = A.max(-1) ** 2
        self.prev_A = A
        return self._solve(A, gamma, mdp, policy)

    def _solve_general(self,
              gamma: float, 
              mdp: MDP,
              policy: npt.NDArray[np.long]) -> BoundResult:
        G = mdp.build_stationary_matrix(policy, gamma=gamma)
        K = mdp.build_K(policy)

        A = np.zeros((mdp.dim_state, mdp.dim_state))
        for i in range(mdp.dim_state):
            e_i = np.zeros(self.dim_state)
            e_i[i] = 1
            self.e_i.value = e_i
            for s in range(mdp.dim_state):
                self.KG.value = K[s] @ G
                if self.prev_theta:
                    self.theta.value = self.prev_theta[i,s]
                res = self.MD_problem.solve(method='dccp', solver = self.solver, ccp_times=self.dim_state * 2, max_iter=self.MAX_ITER)[0]
                A[s,i] = res
                if self.theta.value:
                    self.prev_theta[i,s] = self.theta.value

        A = A.max(-1) ** 2
        self.prev_A = A
        
        return self._solve(A, gamma, mdp, policy)


