import cvxpy as cp
import numpy as np
import dccp
import numpy.typing as npt
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.reward_set import RewardSet, RewardSetPolytope, RewardSetType, RewardSetRewardFree
from typing import NamedTuple, Sequence,List
from multireward_ope.tabular.policy import Policy



class BoundResult(NamedTuple):
    value: float
    w: npt.NDArray[np.float64]

class CharacteristicTimeSolver(object):
    """ Class to solve the characteristic time problem """
    dim_state: int
    dim_actions: int
    theta: Sequence[cp.Variable]
    KG: Sequence[cp.Parameter]
    e_i: cp.Parameter
    rho_problem: Sequence[cp.Problem]
    rewards: Sequence[RewardSet]
    policies: Sequence[Policy]
    solver: str
    vertices: npt.NDArray[np.float64]
    prev_theta: npt.NDArray[np.float64]
    prev_omega: npt.NDArray[np.float64] | None
    prev_A: npt.NDArray[np.float64] | None
    MAX_ITER: int = 500

    def __init__(self, dim_state: int, dim_actions: int, num_policies: int, solver: str = cp.ECOS):
        self.dim_state = dim_state
        self.dim_actions = dim_actions
        self.theta = [cp.Variable(dim_state, nonneg=True) for _ in range(num_policies)]
        self.KG = [cp.Parameter((dim_state, dim_state)) for _ in range(num_policies)]
        self.e_i = cp.Parameter(dim_state)
        self.num_policies = num_policies
        self.solver = solver
        self.prev_theta = np.zeros((num_policies, dim_state, dim_state, dim_state), order='C')
        self.prev_omega = None
        self.prev_A = None

    def build_problem(self, rewards: Sequence[RewardSet], policies: Sequence[Policy]):
        """Build problem to improve speed

        Args:
            rewards (RewardSet): Set of rewards considered
        """
        self.rho_problem = []
        for rid, rews in enumerate(rewards):
            Ks = self.e_i.T @ self.KG[rid] @ self.theta[rid]


            # obj = cp.Maximize(cp.abs(Ks))
            # self.rho_problem = cp.Problem(obj, rewards.get_constraints(self.theta))

            self.rho_problem.append([
                cp.Problem(cp.Maximize(Ks), rews.get_constraints(self.theta[rid])),
                cp.Problem(cp.Maximize(-Ks), rews.get_constraints(self.theta[rid]))])
        self.rewards = rewards
        self.policies = policies

    def _solve_rho(self, solver: str, reward_id: int) -> float:
        v, theta = 0,  None

        # res = self.rho_problem.solve(method='dccp', solver = self.solver, ccp_times=5, max_iter=self.MAX_ITER)[0]
        # return res
        for prob in self.rho_problem[reward_id]:
            vtemp = prob.solve(solver = solver)#, reoptimize=True)
            if vtemp >= v:
                v = vtemp
                #theta = self.theta.value
        return v


    def solve(self,
              gamma: float, 
              mdp: MDP) -> BoundResult:
        """Solve the characteristic time optimization problem

        Args:
            gamma (float): discount facotr
            mdp (MDP): MDP considered
            policies (Sequence[Policy]): list of deterministic policies of size S

        Returns:
            BoundResult: A tuple containing the value of the problem and the optimal solution
        """
        G = [mdp.build_stationary_matrix(policy, gamma=gamma) for policy in self.policies]
        K = [mdp.build_K(policy) for policy in self.policies]

        A = []
        # import pdb
        # pdb.set_trace()
        for r_id, rews in enumerate(self.rewards):
            if self.rewards[0].set_type == RewardSetType.FINITE:
                A.append(self._build_A_finite(mdp, r_id, G[r_id], K[r_id]))
            elif self.rewards[0].set_type == RewardSetType.REWARD_FREE:
                A.append(self._build_A_rewardfree(mdp, G[r_id], K[r_id]))
            elif self.rewards[0].set_type == RewardSetType.NONE:
                raise Exception('RewardsSetType is set to None!')
            else:
                A.append(self._build_A_general(mdp, r_id, G[r_id], K[r_id]))
        # import pdb
        # pdb.set_trace()
        self.prev_A = np.vstack(A)
        return self._solve(self.prev_A, mdp)

    def evaluate(self,
              omega: npt.NDArray[np.float64],
              gamma: float,
              epsilon: float, 
              mdp: MDP,
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
            self.solve(gamma, mdp)
        objs = []
        for pol_id, policy in enumerate(self.policies):
            obj = np.multiply(self.prev_A[pol_id], omega[np.arange(mdp.dim_state), policy]).max()
            obj *= (gamma / (2 * epsilon * (1 -  gamma))) ** 2
            objs.append(obj)
        return objs

        
    def _solve(self, A: npt.NDArray[np.float64], mdp: MDP):
        normalization = 1/np.max(np.abs(A))
        
        omega = cp.Variable((self.dim_state, self.dim_actions), nonneg=True)
        if self.prev_omega is not None:
            omega.value = self.prev_omega

        constraints = [cp.sum(omega) == 1]
        constraints.extend(
            [cp.sum(omega[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], omega)) for s in range(self.dim_state)])
    
        objs= []
        for pol_id, policy in enumerate(self.policies):
            omega_pi = omega[np.arange(mdp.dim_state), policy]
            obj = cp.multiply(A[pol_id]* normalization, cp.inv_pos(omega_pi))
            objs.append(cp.max(obj))

        objs = cp.vstack(objs)
        obj = cp.Minimize(cp.max(objs))
        T_problem = cp.Problem(obj, constraints)
        try:
            res = T_problem.solve(solver=cp.CLARABEL, verbose=False)
        except:
            res = T_problem.solve(solver=cp.ECOS, verbose=False)
   
        self.prev_omega = omega.value
        return BoundResult(res / normalization, omega.value)

        
    def _build_A_rewardfree(self,
              mdp: MDP,
              G: np.ndarray,
              K: np.ndarray) -> np.ndarray:

        A = np.zeros((mdp.dim_state, mdp.dim_state))
        for i in range(mdp.dim_state):
            for s in range(mdp.dim_state):
                KG = K[s] @ G
                Ai_s = KG[i]
                pos_idxs = Ai_s >= 0
                
                A[s,i] = np.maximum(Ai_s[pos_idxs].sum(-1), Ai_s[~pos_idxs].sum(-1))
        A = A.max(-1) ** 2
        
        return A

    def _build_A_general(self, 
              mdp: MDP,
              rewardset_id: int,
              G: np.ndarray,
              K: np.ndarray) -> np.ndarray:

        A = np.zeros((mdp.dim_state, mdp.dim_state))
        for i in range(mdp.dim_state):
            e_i = np.zeros(self.dim_state)
            e_i[i] = 1
            self.e_i.value = e_i
            for s in range(mdp.dim_state):
                self.KG[rewardset_id].value = K[s] @ G
                if self.prev_theta is not None:
                    self.theta[rewardset_id].value = self.prev_theta[rewardset_id,i,s]
                res = self._solve_rho(solver = self.solver, reward_id=rewardset_id)
                A[s,i] = res
                if self.theta[rewardset_id].value is not None:
                    self.prev_theta[rewardset_id, i,s] = self.theta[rewardset_id].value

        A = A.max(-1) ** 2
        return A


    def _build_A_finite(self,
              mdp: MDP,
              rewardset_id: int,
              G: np.ndarray,
              K: np.ndarray) -> np.ndarray:
        A = np.zeros((mdp.dim_state, mdp.dim_state))

        
        rewards: npt.NDArray[np.float64] = self.rewards[rewardset_id].rewards
        nr = rewards.shape[0]
        for i in range(mdp.dim_state):
            e_i = np.zeros(self.dim_state)
            e_i[i] = 1
            for s in range(mdp.dim_state):
                obj = []
                KG = K[s] @ G
                for r in range(nr):
                    obj.append(np.abs(e_i.T @ KG @ rewards[r]))
                res = np.max(obj)
                A[s,i] = res
        A = A.max(-1) ** 2
        
        return A


