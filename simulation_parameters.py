import numpy as np
import numpy.typing as npt
from typing import NamedTuple, List, Tuple
from enum import Enum
from multireward_ope.tabular.envs.riverswim import RiverSwim
from multireward_ope.tabular.reward_set import RewardSet, RewardSetBox, RewardSetCircle, \
    RewardSetFinite, RewardSetPolytope, RewardSetRewardFree, RewardSetConfig
from multireward_ope.tabular.envs.env import EnvParameters, EnvType
from multireward_ope.tabular.policy import Policy

class SimulationGeneralParameters(NamedTuple):
    num_sims: int
    reward_set: RewardSetConfig
    freq_eval: int
    discount_factor: float
    delta: float
    policy: Policy

class SimulationParameters(NamedTuple):
    env_parameters: EnvParameters
    sim_parameters: SimulationGeneralParameters

class SimulationConfiguration(NamedTuple):
    sim_parameters: SimulationGeneralParameters
    envs: List[Tuple[EnvParameters, List[NamedTuple]]]
