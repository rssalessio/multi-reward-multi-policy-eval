from enum import Enum
from multireward_ope.tabular.agents.base_agent import Agent
from multireward_ope.tabular.agents.mr_nas_pe import MRNaSPE
from multireward_ope.tabular.agents.noisy_policy import NoisyPolicy
from multireward_ope.tabular.agents.gvf_explorer import GVFExplorer
from multireward_ope.tabular.agents.sf_nr import SFNR
from multireward_ope.tabular.dataclasses import AgentParameters
from multireward_ope.tabular.agents.dataclasses import AgentType


def make_agent(cfg: AgentParameters, **kwargs) -> Agent:
    match cfg.type:
        case AgentType.NOISY_POLICY:
            return NoisyPolicy(agent_params=cfg.parameters, **kwargs)
        case AgentType.GVFEXplorer:
            return GVFExplorer(cfg=cfg.parameters, **kwargs)
        case AgentType.MR_NAS_PE:
            return MRNaSPE(cfg=cfg.parameters, **kwargs)
        case AgentType.SF_NR:
            return SFNR(cfg=cfg.parameters, **kwargs)
        case _:
            raise NotImplementedError(f'Type {cfg.type} not found.')