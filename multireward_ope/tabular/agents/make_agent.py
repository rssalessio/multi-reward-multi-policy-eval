from enum import Enum
from multireward_ope.tabular.agents.base_agent import AgentParameters, Agent
from multireward_ope.tabular.agents.mr_nas_pe import MRNaSPE, MRNaSPEParameters
from multireward_ope.tabular.agents.noisy_policy import NoisyPolicy, NoisyPolicyParameters, PolicyNoiseType
from multireward_ope.tabular.reward_set import RewardSet
from multireward_ope.tabular.policy import Policy

class AgentType(Enum):
    NOISY_POLICY = 'Noisy Policy'
    MR_NAS_PE = 'MR-NaS-PE'

def make_agent(agent_parameters: AgentParameters, policy: Policy, reward_set: RewardSet) -> Agent:
    if isinstance(agent_parameters, MRNaSPEParameters):
        return MRNaSPE(agent_parameters, policy, reward_set)
    elif isinstance(agent_parameters, NoisyPolicyParameters):
        return NoisyPolicy(agent_parameters, policy, reward_set)
    else:
        raise NotImplementedError(f'Type {agent_parameters.__str__} not found.')