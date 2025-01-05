from multireward_ope.continuous.envs.deepsea import DeepSea
from multireward_ope.continuous.envs.dataclasses import EnvParameters, EnvType


def make_env(env: EnvParameters):
    match env.type:
        case EnvType.DEEP_SEA.value:
            return DeepSea(env.parameters)
        case _:
            raise NotImplementedError(f'Type {env.type} not found.')