from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from multireward_ope.tabular.agents.mr_nas_pe import MRNaSPEConfig
from multireward_ope.tabular.agents.noisy_policy import  NoisyPolicyParameters
from multireward_ope.tabular.agents.gvf_explorer import GVFExplorerConfig

class AgentType(str, Enum):
    NOISY_POLICY = 'Noisy-Policy'
    MR_NAS_PE = 'MR-NaS-PE'
    GVFExplorer = 'GVFExplorer'

@dataclass
class AgentParameters:
    type: AgentType
    parameters: NoisyPolicyParameters | GVFExplorerConfig | MRNaSPEConfig


    @staticmethod
    def name(cls: AgentParameters) -> str:
        match cls.type:
            case AgentType.NOISY_POLICY:
                params = NoisyPolicyParameters.name(cls.parameters)
            case AgentType.GVFExplorer:
                params = GVFExplorerConfig.name(cls.parameters)
            case AgentType.MR_NAS_PE:
                params = MRNaSPEConfig.name(cls.parameters)
            case _:
                raise Exception(f'Type {cls.type} not found')
        return f'{cls.type}_{params}'
