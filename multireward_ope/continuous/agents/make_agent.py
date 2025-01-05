import torch
from enum import Enum
from multireward_ope.continuous.agents.agent import Agent
from multireward_ope.continuous.agents.dbmr_pe.dbmr_pe import DBMRPE
from multireward_ope.continuous.dataclasses import AgentParameters
from multireward_ope.continuous.agents.dataclasses import AgentType


def make_agent(dim_state: int, num_actions: int, num_rewards: int, horizon: int, cfg: AgentParameters, device: torch.device) -> Agent:
    match cfg.type:
        case AgentType.DBMR_PE:
            return DBMRPE.make_default_agent(dim_state, num_actions, num_rewards, horizon, cfg.parameters, device)
        case _:
            raise NotImplementedError(f'Type {cfg.type} not found.')