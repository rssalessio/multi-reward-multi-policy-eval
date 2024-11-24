from __future__ import annotations
import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Callable
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.characteristic_time import BoundResult, CharacteristicTimeSolver
from multireward_ope.tabular.reward_set import RewardSet

# Define a named tuple to store experience data
class Experience(NamedTuple):
    s_t: int     # State at time t
    a_t: int     # Action at time t
    s_tp1: int   # State at time t+1

# Define a named tuple for agent parameters
class AgentParameters(NamedTuple):
    dim_state_space: int        # Dimension of state space
    dim_action_space: int       # Dimension of action space
    discount_factor: float      # Discount factor for future rewards
    horizon: int                # Horizon (time steps) to plan ahead
    frequency_evaluation: int   # How often to evaluate the transition function
    delta: float                # confidence
    epsilon: float              # accuracy
    solver_type: str            # solver value

# Define an abstract agent class
class Agent(ABC):
    # Class attributes
    dim_state_space: int
    dim_action_space: int
    discount_factor: float
    exp_visits: npt.NDArray[np.float64]
    total_state_visits: npt.NDArray[np.float64]
    last_visit: npt.NDArray[np.float64]
    greedy_policy: npt.NDArray[np.int64]
    omega: npt.NDArray[np.float64]
    horizon: int
    delta: float
    epsilon: float
    solver: CharacteristicTimeSolver
    policy: npt.NDArray[np.long]
    solver_type: str

    # Initialize the agent with agent parameters
    def __init__(self, agent_parameters: AgentParameters,  policy: npt.NDArray[np.long], rewards: RewardSet):
        self.dim_state_space = agent_parameters.dim_state_space
        self.dim_action_space = agent_parameters.dim_action_space
        self.discount_factor = agent_parameters.discount_factor
        self.exp_visits = np.zeros((self.ns, self.na, self.ns), order='C')
        self.state_action_visits = np.zeros((self.ns, self.na), order='C')
        self.total_state_visits = np.zeros((self.ns), order='C')
        self.last_visit = np.zeros((self.ns), order='C')
        self.omega = np.ones((self.ns, self.na), order='C')
        self.exploration_parameter = self.suggested_exploration_parameter(self.ns, self.na)
        self.horizon = agent_parameters.horizon
        self.frequency_evaluation = agent_parameters.frequency_evaluation
        self.delta = agent_parameters.delta
        self.solver_type = agent_parameters.solver_type
        self.solver = CharacteristicTimeSolver(self.ns, self.na, solver=self.solver_type)
        self.solver.build_problem(rewards)
        self.policy = np.zeros((self.ns, self.na))
        self.policy[np.arange(self.ns), policy] = 1.
        self.epsilon = agent_parameters.epsilon

    @property
    def beta(self):
        return self._beta(self.state_action_visits)
    
    @property
    def mdp(self) -> MDP:
        return MDP(P = self.empirical_transition())
    
    def _beta(self, n: float) -> float:
        c1 = np.log(1 / self.delta)
        c2 = np.log(np.e * (1 + n/(self.ns - 1)))
        return c1 + (self.ns - 1) * c2.sum()
    
    @property
    def Z_t(self) -> float:
        t = self.state_action_visits.sum()
        return t / self.U_t
    
    @property
    def U_t(self) -> float:
        w = self.state_action_visits / self.state_action_visits.sum()
        mdp = MDP(P = self.empirical_transition())
        return self.solver.evaluate(
            omega=self.omega,
            gamma=self.discount_factor,
            epsilon=self.epsilon,
            mdp=mdp,
            policy=self.policy
        )

    # Property getter for state space dimension
    @property
    def ns(self) -> int:
        return self.dim_state_space

    # Property getter for action space dimension
    @property
    def na(self) -> int:
        return self.dim_action_space

    # Abstract static method to return the suggested exploration parameter
    @abstractmethod
    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return 1.

    # Method to compute forced exploration probability
    def forced_exploration_callable(self, state: int, step: int, minimum_exploration: float = 1e-3) -> float:
        return max(minimum_exploration, 1 / max(1, self.total_state_visits[state]) ** self.exploration_parameter)

    # Abstract method for forward pass
    @abstractmethod
    def forward(self, state: int, step: int) -> int:
        raise NotImplementedError

    # Abstract method for processing experience
    @abstractmethod
    def process_experience(self, experience: Experience, step: int) -> None:
        raise NotImplementedError

    # Method for backward pass (update agent)
    def backward(self, experience: Experience, step: int) -> None:
        # Increment visit count for the current state-action pair
        self.exp_visits[experience.s_t, experience.a_t, experience.s_tp1] += 1
        self.state_action_visits[experience.s_t, experience.a_t] += 1
        
        # Update last visit time and total state visits count for the next state
        self.last_visit[experience.s_tp1] = step + 1
        self.total_state_visits[experience.s_tp1] += 1
        
        # If this is the first time step, update last visit time and total state visits count for the current state
        if step == 0:
            self.last_visit[experience.s_t] = step
            self.total_state_visits[experience.s_t] += 1

        # Process the experience to update the agent's internal model
        self.process_experience(experience, step)

    def empirical_transition(self, prior_p: float = 1.0) -> npt.NDArray[np.float64]:
        prior_transition = prior_p * np.ones((self.ns, self.na, self.ns))
        posterior_transition = prior_transition + self.exp_visits

        # Compute MLE of the parameters
        P = posterior_transition / posterior_transition.sum(-1, keepdims=True)
        return P