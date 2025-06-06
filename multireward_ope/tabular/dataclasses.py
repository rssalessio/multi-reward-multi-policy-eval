from __future__ import annotations
from dataclasses import dataclass
from multireward_ope.tabular.envs.dataclasses import EnvParameters
from multireward_ope.tabular.agents.dataclasses import AgentParameters
from multireward_ope.tabular.reward_set import RewardSetType


@dataclass
class Config:
    experiment: ExperimentConfig
    agent: AgentParameters
    environment: EnvParameters

    @staticmethod
    def name(cls: Config) -> str:
        pol = 'SinglePolicy' if cls.experiment.single_policy else 'MultiplePolicies'
        folder = f'{cls.experiment.reward_set}_{pol}'

        return f'/{folder}/{ExperimentConfig.name(cls.experiment)}_{EnvParameters.name(cls.environment)}_{AgentParameters.name(cls.agent)}'

@dataclass
class ExperimentConfig:
    num_simulations: int
    num_processes: int
    horizon: int
    frequency_evaluation: int
    delta: float
    epsilon: float
    discount_factor: float
    solver_type: str
    reward_set: RewardSetType
    single_policy: bool

    @staticmethod
    def name(cls) -> str:
        return f'{cls.horizon}'
