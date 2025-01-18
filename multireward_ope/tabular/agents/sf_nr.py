import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from multireward_ope.tabular.agents.base_agent import Agent, Experience
from enum import Enum
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.utils import policy_evaluation

@dataclass
class SFNRConfig:
    alpha: float


    @classmethod
    def name(cls) -> str:
        return ''

    
class SFNR(Agent):
    """ SFNR Algorithm 
        @See https://arxiv.org/pdf/2202.11133
    """

    def __init__(self,
                 cfg: SFNRConfig, **kwargs):
        self.cfg = cfg
        super().__init__(**kwargs)
        self.uniform_policy = np.ones(self.dim_action_space) / self.dim_action_space
        eval_rewards = self.rewards.canonical_rewards()

        self.eval_rewards = np.zeros((eval_rewards.shape[0], self.dim_state_space, self.dim_action_space))
        for i in range(eval_rewards.shape[0]):
            self.eval_rewards[i, np.arange(self.dim_state_space), self.policy.argmax(-1)] = eval_rewards[i]
    
    @property
    def name(self) -> str:
        return f'GVFExplorer'
    
    def forward(self, state: int, step: int) -> int:
        alpha = self.suggested_exploration_parameter(self.dim_state_space, self.dim_action_space)
        
        omega = self.omega[state] / self.omega[state].sum()
        policy = (1-alpha) * omega + alpha * self.uniform_policy
        return np.random.choice(self.na, p=policy)

    def process_experience(self, experience: Experience, step: int) -> None:
        if (step + 1 ) % self.cfg.policy_update_frequency:
            for R in self.eval_rewards:
                mdp = MDP(P=self.empirical_transition())
                hat_values = np.array([
                    policy_evaluation(self.discount_factor, 
                                    mdp.P,
                                    R=self.eval_rewards[r], policy=self.policy.argmax(-1))
                                    for r in range(self.eval_rewards.shape[0])]).T
                P = mdp.P[np.arange(mdp.P.shape[0]), self.policy.argmax(-1)]
                avg_V = P @ hat_values
                var_V =  P @ (hat_values ** 2) - (avg_V) ** 2
                M = var_V.sum(-1)
                import pdb
                pdb.set_trace()
                
            

    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return self.cfg.noise_parameter