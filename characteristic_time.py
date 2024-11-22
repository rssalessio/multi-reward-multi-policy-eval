import cvxpy as cp
import numpy as np
import dccp
import numpy.typing as npt
from enum import Enum
from mdp import MDP
from reward_set import RewardSet, RewardSetCircle
from typing import NamedTuple



class BoundResult(NamedTuple):
    value: float
    w: npt.NDArray[np.float64]

class CharacteristicTimeSolver(object):
    dim_state: int
    dim_actions: int
    theta: cp.Variable
    KG: cp.Parameter
    e_i: cp.Parameter
    MD_problem: cp.Problem

    def __init__(self, dim_state: int, dim_actions: int):
        self.dim_state = dim_state
        self.dim_actions = dim_actions
        self.theta = cp.Variable(dim_state, nonneg=True)
        self.KG = cp.Parameter((dim_state, dim_state))
        self.e_i = cp.Parameter(dim_state)

    def build_problem(self, rewards: RewardSet):
        Ks = self.e_i.T @ self.KG @ self.theta
        obj = cp.Maximize(cp.abs(Ks))
        self.MD_problem = cp.Problem(obj, rewards.get_constraints(self.theta))

    def solve(self,
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
                res = self.MD_problem.solve(method='dccp')[0]
                A[s,i] = res
        
        omega = cp.Variable((self.dim_state, self.dim_actions), nonneg=True)
        A = A.max(-1) ** 2
        
        constraints = [cp.sum(omega) == 1]
        constraints.extend(
            [cp.sum(omega[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], omega)) for s in range(self.dim_state)])
    
        omega_pi = omega[np.arange(mdp.dim_state), policy]

        obj = cp.Minimize(cp.max(cp.multiply(A, cp.inv_pos(omega_pi))))
        T_problem = cp.Problem(obj, constraints)
        res = T_problem.solve()
        return BoundResult(res, omega.value)



if __name__ == '__main__':
    mdp = MDP.generate_random_mdp(3, 2)
    policy = np.array([0, 1, 0], dtype=np.long)
    rewards = RewardSetCircle(mdp.dim_state, np.zeros(mdp.dim_state), radius=1, p=2)

    solver = CharacteristicTimeSolver(mdp.dim_state, mdp.dim_action)
    solver.build_problem(rewards)

    print(solver.solve(0.9, mdp, policy))

    


            
