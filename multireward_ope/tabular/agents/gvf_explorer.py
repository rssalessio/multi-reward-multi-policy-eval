from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from multireward_ope.tabular.agents.base_agent import Agent, Experience
from enum import Enum
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.utils import policy_evaluation
from multireward_ope.tabular.reward_set import RewardSetType
@dataclass
class GVFExplorerConfig:
    noise_parameter: float
    policy_update_frequency: int


    @staticmethod
    def name(cls: GVFExplorerConfig) -> str:
        return f''

    
class GVFExplorer(Agent):
    """ GVFExplorer Algorithm 
        @See https://arxiv.org/pdf/2405.07838
    """

    def __init__(self,
                 cfg: GVFExplorerConfig, **kwargs):
        self.cfg = cfg
        super().__init__(**kwargs)

        assert len(self.policies) > 1 and np.all([rew.set_type == RewardSetType.FINITE for rew in self.rewards]), 'GVFExplorer Only supports multiple policies and finite rewards'

        self.uniform_policy = np.ones(self.dim_action_space) / self.dim_action_space
        self.M = np.ones((self.num_policies, self.dim_state_space))
    
    @property
    def name(self) -> str:
        return f'GVFExplorer'
    
    def forward(self, state: int, step: int) -> int:
        alpha = self.suggested_exploration_parameter(self.dim_state_space, self.dim_action_space)

        policy = np.sqrt((self.policies[:, state] * self.M[:,state, None]).sum(0)) + 1e-7
        policy = policy / policy.sum()
        policy = (1-alpha) * policy + alpha * self.uniform_policy
        return np.random.choice(self.na, p=policy)

    def process_experience(self, experience: Experience, step: int) -> None:
        if (step + 1 ) % self.cfg.policy_update_frequency == 0:
            for rid, reward_set in enumerate(self.rewards):
                R = reward_set.rewards
                mdp = MDP(P=self.empirical_transition())
                pol = self.policies[rid].argmax(-1)
                rew = np.zeros((self.dim_state_space, self.dim_action_space))
                rew[np.arange(self.dim_state_space), pol] = R
                hat_values = policy_evaluation(self.discount_factor, 
                                    mdp.P,
                                    R=rew, policy=pol)
                #P = mdp.P[np.arange(mdp.P.shape[0]), pol]
                P = mdp.P.reshape(-1, mdp.dim_state)

                avg_V = P @ hat_values
                var_V =  P @ (hat_values ** 2) - (avg_V) ** 2
                rew_var = ((self.discount_factor ** 2) * var_V).reshape(mdp.dim_state,-1)
                var_G = policy_evaluation(self.discount_factor**2, mdp.P, rew_var, policy=pol)
                
                self.M[rid] = var_G
                
            

    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return self.cfg.noise_parameter