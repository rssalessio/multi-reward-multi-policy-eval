from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from multireward_ope.continuous.agents.dbmr_pe.config import DBMRPEConfig

class AgentType(str, Enum):
    DBMR_PE = 'DBMR-PE'

@dataclass
class AgentParameters:
    type: AgentType
    parameters: DBMRPEConfig

    @staticmethod
    def name(cls: AgentParameters) -> str:
        match cls.type:
            case AgentType.DBMR_PE:
                params = DBMRPEConfig.name(cls.parameters)
            case _:
                raise Exception(f'Type {cls.type} not found')
        return f'{cls.type}_{params}'

    @staticmethod
    def short_name(cls: AgentParameters) -> str:
        match cls.type:
            case AgentType.DBMR_PE:
                return cls.type
            case _:
                raise Exception(f'Type {cls.type} not found')
