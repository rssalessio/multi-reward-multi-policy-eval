import numpy as np
import cvxpy as cp
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.agents.base_agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
from multireward_ope.tabular.characteristic_time import BoundResult, CharacteristicTimeSolver
from multireward_ope.tabular.reward_set import RewardSet
from enum import Enum
import numpy.typing as npt

class PolicyNoiseType(Enum):
    UNIFORM = 'Uniform'
    VISITATION = 'Visitation based'

class NoisyPolicyParameters(NamedTuple):
    agent_parameters: AgentParameters
    noise_type: PolicyNoiseType
    noise_parameter: float

    @property
    def name(self) -> str:
        return f'Noisy policy ({self.noise_type.__str__})'


class NoisyPolicy(Agent):
    """ NoisyPolicy Algorithm """

    def __init__(self, parameters: NoisyPolicyParameters, policy: npt.NDArray[np.long], rewards: RewardSet):
        super().__init__(parameters.agent_parameters, policy, rewards)
        self.parameters = parameters
        self.uniform_policy = np.ones(self.dim_action_space) / self.dim_action_space
    

    def forward(self, state: int, step: int) -> int:
        alpha = self.suggested_exploration_parameter()
        if self.parameters.noise_type == PolicyNoiseType.UNIFORM:
            policy = (1 - alpha) * self.policy + alpha * self.uniform_policy
        elif self.parameters.noise_type == PolicyNoiseType.VISITATION:
            act_visit = self.state_action_visits[state] / self.state_action_visits[state].sum()
            policy = (1 - alpha) * self.policy + alpha * act_visit
        return np.random.choice(self.na, p=policy)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        pass

    def suggested_exploration_parameter(self) -> float:
        return self.parameters.noise_parameter