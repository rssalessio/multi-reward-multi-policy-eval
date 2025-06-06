from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from multireward_ope.tabular.agents.base_agent import Agent, Experience
from enum import Enum

class PolicyNoiseType(str, Enum):
    UNIFORM = 'Uniform'
    VISITATION = 'Visitation based'

@dataclass
class NoisyPolicyParameters:
    noise_type: PolicyNoiseType
    noise_parameter: float

    @staticmethod
    def name(cls: NoisyPolicyParameters) -> str:
        return f'{cls.noise_type}_{cls.noise_parameter}'

    
class NoisyPolicy(Agent):
    """ NoisyPolicy Algorithm """

    def __init__(self,
                 agent_params: NoisyPolicyParameters, **kwargs):
        self.parameters = agent_params
        super().__init__(**kwargs)
        self.uniform_policy = np.ones(self.dim_action_space) / self.dim_action_space
    
    @property
    def name(self) -> str:
        return f'Noisy Policy ({self.parameters.noise_type})'
    
    def forward(self, state: int, step: int) -> int:
        alpha = self.suggested_exploration_parameter(self.dim_state_space, self.dim_action_space)

        policy = self.mixture_policy[state]
        
        if self.parameters.noise_type == PolicyNoiseType.UNIFORM.value:
            policy = (1 - alpha) * policy + alpha * self.uniform_policy
        elif self.parameters.noise_type == PolicyNoiseType.VISITATION.value:
            visits = np.maximum(1, self.state_action_visits[state])
            exp_visits = -visits + visits.max() + 1
            act_visit = np.exp(np.log(exp_visits) - np.log(np.sum(exp_visits)))
            policy = (1 - alpha) * policy + alpha * act_visit
        return np.random.choice(self.na, p=policy)

    def process_experience(self, experience: Experience, step: int) -> None:
        pass

    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return self.parameters.noise_parameter