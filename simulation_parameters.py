import numpy as np
import numpy.typing as npt
from typing import NamedTuple, List, Tuple
from enum import Enum
from multireward_ope.tabular.envs.riverswim import RiverSwim
from multireward_ope.tabular.reward_set import RewardSet, RewardSetCircle, \
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

class Results(NamedTuple):
    step: int
    omega: npt.NDArray[np.float64]
    total_state_visits: npt.NDArray[np.float64]
    last_visit: npt.NDArray[np.float64]
    exp_visits: npt.NDArray[np.float64]
    V_res: npt.NDArray[np.float64]
    Q_res: npt.NDArray[np.float64]
    pi_res: Policy
    elapsed_time: float