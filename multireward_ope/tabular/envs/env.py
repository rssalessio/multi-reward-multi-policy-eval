from typing import NamedTuple
from enum import Enum
from multireward_ope.tabular.envs.riverswim import RiverSwim, RiverSwimParameters

class EnvType(Enum):
    RIVERSWIM = 'Riverswim'
    FORKED_RIVERSWIM = 'ForkedRiverswim'
    DOUBLE_CHAIN = 'DoubleChain'
    N_ARMS = 'NArms'


class EnvParameters(NamedTuple):
    env_type: EnvType
    parameters: RiverSwimParameters # | DoubleChainParameters | NArmsParameters | ForkedRiverSwimParameters
    horizon: int

def make_env(env: EnvParameters):
    match env.env_type:
        case EnvType.RIVERSWIM:
            return RiverSwim(num_states=env.parameters.num_states)
        # case EnvType.DOUBLE_CHAIN:
        #     return DoubleChain(length = env.parameters.length, p = env.parameters.p)
        # case EnvType.N_ARMS:
        #     return NArms(num_arms = env.parameters.num_arms, p0 = env.parameters.p0)
        # case EnvType.FORKED_RIVERSWIM:
        #     return ForkedRiverSwim(river_length = env.parameters.river_length)
        case _:
            raise NotImplementedError(f'Type {env.env_type.value} not found.')