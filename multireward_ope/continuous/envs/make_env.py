from multireward_ope.tabular.envs.riverswim import RiverSwim
from multireward_ope.tabular.envs.doublechain import DoubleChain
from multireward_ope.tabular.envs.narms import NArms
from multireward_ope.tabular.envs.forked_riverswim import ForkedRiverSwim
from multireward_ope.tabular.envs.dataclasses import EnvParameters, EnvType


def make_env(env: EnvParameters):
    match env.type:
        case EnvType.RIVERSWIM.value:
            return RiverSwim(env.parameters)
        case EnvType.DOUBLE_CHAIN.value:
            return DoubleChain(env.parameters)
        case EnvType.N_ARMS:
            return NArms(env.parameters)
        case EnvType.FORKED_RIVERSWIM:
            return ForkedRiverSwim(env.parameters)
        case _:
            raise NotImplementedError(f'Type {env.type} not found.')