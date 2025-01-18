from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from multireward_ope.continuous.agents.dbmr_pe.config import DBMRPEConfig
from multireward_ope.continuous.agents.uniform.config import UniformConfig
from multireward_ope.continuous.agents.rnd_pe.config import RNDPEConfig
class AgentType(str, Enum):
    DBMR_PE = 'DBMR-PE'
    UNIFORM = 'UNIFORM'
    RND_PE = 'RND-PE'

@dataclass
class AgentParameters:
    type: AgentType
    parameters: DBMRPEConfig | UniformConfig

    @staticmethod
    def name(cls: AgentParameters) -> str:
        match cls.type:
            case AgentType.DBMR_PE:
                params = DBMRPEConfig.name(cls.parameters)
            case AgentType.UNIFORM:
                params = UniformConfig.name(cls.parameters)
            case AgentType.RND_PE:
                params = RNDPEConfig.name(cls.parameters)
            case _:
                raise Exception(f'Type {cls.type} not found')
        return f'{cls.type}_{params}'

    @staticmethod
    def short_name(cls: AgentParameters) -> str:
        match cls.type:
            case AgentType.DBMR_PE:
                return cls.type
            case AgentType.UNIFORM:
                return cls.type
            case AgentType.RND_PE:
                return cls.type
            case _:
                raise Exception(f'Type {cls.type} not found')
