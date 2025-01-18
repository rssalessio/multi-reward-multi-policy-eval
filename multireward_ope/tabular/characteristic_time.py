import cvxpy as cp
import numpy as np
import dccp
import numpy.typing as npt
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.reward_set import RewardSet, RewardSetPolytope, RewardSetType, RewardSetRewardFree
from typing import NamedTuple, Sequence
from multireward_ope.tabular.policy import Policy



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
    rho_problem: cp.Problem
    rewards: RewardSet
    solver: str
    vertices: npt.NDArray[np.float64]
    prev_theta: npt.NDArray[np.float64]
    prev_omega: npt.NDArray[np.float64] | None
    prev_A: npt.NDArray[np.float64] | None
    MAX_ITER: int = 500

    def __init__(self, dim_state: int, dim_actions: int, num_policies: int, solver: str = cp.ECOS):
        self.dim_state = dim_state
        self.dim_actions = dim_actions
        self.theta = cp.Variable(dim_state, nonneg=True)
        self.KG = cp.Parameter((dim_state, dim_state))
        self.e_i = cp.Parameter(dim_state)
        self.num_policies = num_policies
        self.solver = solver
        self.prev_theta = np.zeros((num_policies, dim_state, dim_state, dim_state), order='C')
        self.prev_omega = None
        self.prev_A = None

    def build_problem(self, rewards: RewardSet):
        """Build problem to improve speed

        Args:
            rewards (RewardSet): Set of rewards considered
        """
        Ks = self.e_i.T @ self.KG @ self.theta


        # obj = cp.Maximize(cp.abs(Ks))
        # self.rho_problem = cp.Problem(obj, rewards.get_constraints(self.theta))

        self.rho_problem = [
            cp.Problem(cp.Maximize(Ks), rewards.get_constraints(self.theta)),
            cp.Problem(cp.Maximize(-Ks), rewards.get_constraints(self.theta))]
        self.rewards = rewards

    def _solve_rho(self, solver: str) -> float:
        v, theta = 0,  None

        # res = self.rho_problem.solve(method='dccp', solver = self.solver, ccp_times=5, max_iter=self.MAX_ITER)[0]
        # return res
        for prob in self.rho_problem:
            vtemp = prob.solve(solver = solver)#, reoptimize=True)
            if vtemp >= v:
                v = vtemp
                theta = self.theta.value
        return v


    def solve(self,
              gamma: float, 
              mdp: MDP,
              policies: Sequence[Policy]) -> BoundResult:
        """Solve the characteristic time optimization problem

        Args:
            gamma (float): discount facotr
            mdp (MDP): MDP considered
            policies (Sequence[Policy]): list of deterministic policies of size S

        Returns:
            BoundResult: A tuple containing the value of the problem and the optimal solution
        """
        if self.rewards.set_type == RewardSetType.FINITE:
            return self._solve_finite(gamma, mdp, policies)
        elif self.rewards.set_type == RewardSetType.REWARD_FREE:
            return self._solve_rewardfree(gamma, mdp, policies)
        elif self.rewards.set_type == RewardSetType.NONE:
            raise Exception('RewardsSetType is set to None!')
        else:
            return self._solve_general(gamma, mdp, policies)
        

    def evaluate(self,
              omega: npt.NDArray[np.float64],
              gamma: float,
              epsilon: float, 
              mdp: MDP,
              policies: Sequence[Policy],
              force: bool = False) -> Sequence[float]:
        """Solve the characteristic time optimization problem

        Args:
            omega (npt.NDArray[np.float64]): state-action distribution to evaluate
            gamma (float): discount factor
            epsilon (float): accuracy level
            mdp (MDP): MDP considered
            policies (Policy): deterministic policy of size S

        Returns:
            Sequence[float]: A list of values for each policy
        """
        if self.prev_A is None or force:
            self.solve(gamma, mdp, policies=policies)
        objs = []
        for policy in policies:
            obj = np.multiply(self.prev_A, omega[np.arange(mdp.dim_state), policy]).max()
            obj *= (gamma / (2 * epsilon * (1 -  gamma))) ** 2
            objs.append(obj)
        return objs

        
    def _solve(self, A: npt.NDArray[np.float64], gamma: float, mdp: MDP, policies: Sequence[Policy]):
        normalization = (1 - gamma) ** 3
        omega = cp.Variable((self.dim_state, self.dim_actions), nonneg=True)
        if self.prev_omega is not None:
            omega.value = self.prev_omega

        constraints = [cp.sum(omega) == 1]
        constraints.extend(
            [cp.sum(omega[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], omega)) for s in range(self.dim_state)])
    
        objs= []
        for policy in policies:
            omega_pi = omega[np.arange(mdp.dim_state), policy]
            obj = cp.multiply(A* normalization, cp.inv_pos(omega_pi))
            objs.append(cp.max(obj))

        objs = cp.vstack(objs)
        obj = cp.Minimize(cp.max(objs))
        T_problem = cp.Problem(obj, constraints)
        res = T_problem.solve(solver=self.solver)
        self.prev_omega = omega.value
        return BoundResult(res / normalization, omega.value)

        
    def _solve_rewardfree(self,
              gamma: float, 
              mdp: MDP,
              policies: Sequence[Policy]) -> BoundResult:
        G = [mdp.build_stationary_matrix(policy, gamma=gamma) for policy in policies]
        K = [mdp.build_K(policy) for policy in policies]

        A = np.zeros((len(policies), mdp.dim_state, mdp.dim_state))
        for p in range(len(policies)):
            for i in range(mdp.dim_state):
                for s in range(mdp.dim_state):
                    KG = K[p][s] @ G[p]
                    Ai_s = KG[i]
                    pos_idxs = Ai_s >= 0
                    
                    A[p,s,i] = np.maximum(Ai_s[pos_idxs].sum(-1), Ai_s[~pos_idxs].sum(-1))
        A = A.max(-1) ** 2
        self.prev_A = A
        return self._solve(A, gamma, mdp, policies)

    def _solve_general(self,
              gamma: float, 
              mdp: MDP,
              policies: Sequence[Policy]) -> BoundResult:
        G = [mdp.build_stationary_matrix(policy, gamma=gamma) for policy in policies]
        K = [mdp.build_K(policy) for policy in policies]

        A = np.zeros((len(policies), mdp.dim_state, mdp.dim_state))
        for p in range(len(policies)):
            for i in range(mdp.dim_state):
                e_i = np.zeros(self.dim_state)
                e_i[i] = 1
                self.e_i.value = e_i
                for s in range(mdp.dim_state):
                    self.KG.value = K[p][s] @ G[p]
                    if self.prev_theta is not None:
                        self.theta.value = self.prev_theta[p,i,s]
                    res = self._solve_rho(solver = self.solver)
                    A[p, s,i] = res
                    if self.theta.value is not None:
                        self.prev_theta[p, i,s] = self.theta.value

        A = A.max(-1) ** 2
        self.prev_A = A
        
        return self._solve(A, gamma, mdp, policies)


    def _solve_finite(self,
              gamma: float, 
              mdp: MDP,
              policies: Sequence[Policy]) -> BoundResult:
        G = [mdp.build_stationary_matrix(policy, gamma=gamma) for policy in policies]
        K = [mdp.build_K(policy) for policy in policies]
        A = np.zeros((len(policies), mdp.dim_state, mdp.dim_state))

        rewards: npt.NDArray[np.float64] = self.rewards.rewards
        nr = rewards.shape[0]
        for p in range(len(policies)):
            for i in range(mdp.dim_state):
                e_i = np.zeros(self.dim_state)
                e_i[i] = 1
                for s in range(mdp.dim_state):
                    obj = []
                    KG = K[p][s] @ G[p]
                    for r in range(nr):
                        obj.append(np.abs(e_i.T @ KG @ rewards[r]))
                    res = np.max(obj)
                    A[p, s,i] = res
        A = A.max(-1) ** 2
        self.prev_A = A
        
        return self._solve(A, gamma, mdp, policies)


