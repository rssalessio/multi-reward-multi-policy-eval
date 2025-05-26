from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from multireward_ope.tabular.agents.base_agent import Agent, Experience
from multireward_ope.tabular.reward_set import RewardSetType
from enum import Enum
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.utils import policy_evaluation

@dataclass
class SFNRConfig:
    discount_psi: float
    temperature: float

    @staticmethod
    def name(cls: SFNRConfig) -> str:
        return f''

    
class SFNR(Agent):
    """ SFNR Algorithm 
        @See https://arxiv.org/pdf/2202.11133
    """

    def __init__(self,
                 cfg: SFNRConfig, **kwargs):
        self.cfg = cfg
        super().__init__(**kwargs)
        assert len(self.policies) > 1, 'SFNR Only supports multiple policies'
        self.uniform_policy = np.ones(self.dim_action_space) / self.dim_action_space
        
        
        self.discount_psi = cfg.discount_psi
        self.behavior_policy = np.ones((self.dim_state_space, self.dim_action_space)) / self.dim_action_space
        self.psi = np.ones((self.num_policies, self.dim_state_space, self.dim_action_space))

    @property
    def name(self) -> str:
        return f'SF-NR'
    
    def forward(self, state: int, step: int) -> int:
        x= self.behavior_policy[state] / self.cfg.temperature
        eps = 1/np.maximum(1,self.total_state_visits[state].sum())
        if np.random.rand() < eps:
            return np.random.choice(self.dim_action_space)

        omega = np.exp(x - np.max(x))  # Subtracting max for numerical stability
        policy= omega / omega.sum(axis=0) 

        return np.random.choice(self.na, p=policy)

    def process_experience(self, experience: Experience, step: int) -> None:
        alpha_t = 1/self.state_action_visits[experience.s_t, experience.a_t]
        prev_psi = self.psi.copy()
    
        for p in range(self.num_policies):
            tgt = self.discount_psi * self.psi[p, experience.s_tp1, self.policies[p, experience.s_tp1].argmax()]
            delta = 1+ tgt -self.psi[p, experience.s_t, experience.a_t]
            self.psi[p, experience.s_t, experience.a_t] += alpha_t * delta
        delta_psi = np.abs((self.psi - prev_psi).reshape(self.num_policies, -1)).sum(-1).mean()


        tgt = self.discount_factor * self.behavior_policy[experience.s_tp1].max()
        delta = delta_psi + tgt - self.behavior_policy[experience.s_t, experience.a_t]

        self.behavior_policy[experience.s_t, experience.a_t] += alpha_t * delta
            

    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return 1