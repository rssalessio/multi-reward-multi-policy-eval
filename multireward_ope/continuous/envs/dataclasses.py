from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from multireward_ope.continuous.envs.deepsea import DeepSeaConfig

class EnvType(str,Enum):
    DEEP_SEA = 'DeepSea'

@dataclass
class EnvParameters:
    type: EnvType
    parameters: DeepSeaConfig

    @staticmethod
    def name(cls: EnvParameters) -> str:
        match cls.type:
            case EnvType.DEEP_SEA:
                params = DeepSeaConfig.name(cls.parameters)
            case _:
                raise Exception(f'Type {cls.type} not found')
        return f'{cls.type}_{params}'