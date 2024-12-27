from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from multireward_ope.tabular.envs.riverswim import RiverSwimConfig
from multireward_ope.tabular.envs.doublechain import DoubleChainConfig
from multireward_ope.tabular.envs.narms import NArmsConfig
from multireward_ope.tabular.envs.forked_riverswim import ForkedRiverSwimConfig


class EnvType(str,Enum):
    RIVERSWIM = 'Riverswim'
    FORKED_RIVERSWIM = 'ForkedRiverswim'
    DOUBLE_CHAIN = 'DoubleChain'
    N_ARMS = 'NArms'

@dataclass
class EnvParameters:
    type: EnvType
    parameters: RiverSwimConfig | DoubleChainConfig | NArmsConfig | ForkedRiverSwimConfig
    
    @staticmethod
    def name(cls: EnvParameters) -> str:
        match cls.type:
            case EnvType.RIVERSWIM:
                params = RiverSwimConfig.name(cls.parameters)
            case EnvType.DOUBLE_CHAIN:
                params = DoubleChainConfig.name(cls.parameters)
            case EnvType.N_ARMS:
                params = NArmsConfig.name(cls.parameters)
            case EnvType.FORKED_RIVERSWIM:
                params = ForkedRiverSwimConfig.name(cls.parameters)
            case _:
                raise Exception(f'Type {cls.type} not found')
        return f'{cls.type}_{params}'