# This file contains the parameters used in the simulations

from typing import NamedTuple, List
from simulation_parameters import SimulationConfiguration, SimulationGeneralParameters, EnvType, SimulationParameters, EnvParameters, DoubleChainParameters, RiverSwimParameters, make_env, NArmsParameters, ForkedRiverSwimParameters
from multireward_ope.tabular.agents.mr_nas_pe import MRNaSPEParameters
from multireward_ope.tabular.agents.noisy_policy import NoisyPolicyParameters


CONFIG = SimulationConfiguration(
    sim_parameters=SimulationGeneralParameters(
        num_sims=100,
        num_rewards=30,
        freq_eval=500,
        discount_factor=0.9,
        delta=1e-2
    ),
    envs = [
     (
      EnvParameters(EnvType.RIVERSWIM, RiverSwimParameters(num_states=10), 50000),
      [ 
        NoisyPolicyParameters(agent_parameters=None, xi=None),
      ]
     ),
    ]
)