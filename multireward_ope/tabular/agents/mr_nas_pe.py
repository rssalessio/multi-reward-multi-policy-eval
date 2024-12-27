from __future__ import annotations
import numpy as np
import cvxpy as cp
from multireward_ope.tabular.mdp import MDP
from multireward_ope.tabular.agents.base_agent import Agent, Experience
from typing import NamedTuple, Optional
from multireward_ope.tabular.characteristic_time import BoundResult, CharacteristicTimeSolver
from multireward_ope.tabular.reward_set import RewardSet
from enum import Enum
from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class MRNaSPEConfig:
    policy_update_frequency: int
    alpha: float = 0.99
    beta: float = 0.01
    averaged: bool = True

    @staticmethod
    def name(cls: MRNaSPEConfig) -> str:
        return f'{cls.policy_update_frequency}_{cls.averaged}'

class MRNaSPE(Agent):
    """ MR-NaS-PE Algorithm """

    def __init__(self, cfg: MRNaSPEConfig, **kwargs):
        self.alpha_exp = cfg.alpha
        self.beta_exp = cfg.beta
        super().__init__(**kwargs)
        self.cfg = cfg
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.updates = 1
        assert 0 < self.alpha_exp + self.beta_exp <=1, "alpha+beta must be in (0,1]"

    @property
    def name(self) -> str:
        return f'MR-NaS-PE - (Averaged: {self.cfg.averaged})'
 
    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return self.alpha_exp
    
    def compute_exp(self, state: int, step:int) -> float:
        Ns = max(1, self.total_state_visits[state])
        Nsa = self.state_action_visits[state]
        beta = Ns * self.beta_exp * np.log(Ns)
        beta =  beta / max(1,np.max(Nsa - Nsa.min()))
        
        # step 1
        log_numerators = -beta * (Nsa / Ns)
        num_min = np.max(log_numerators)
        log_denominator = num_min + np.log(np.sum(np.exp(log_numerators - num_min)))
        log_expression = log_numerators - log_denominator

        return np.exp(log_expression)

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step, minimum_exploration=1e-3)
        exp_policy =  self.compute_exp(state, step)
        omega = (1-epsilon) * self.omega + epsilon * exp_policy
        omega = omega[state] / omega[state].sum()
        try:
            act =  np.random.choice(self.na, p=omega)
        except:
            raise Exception(f'Error, cant compute probability with {omega}')
        return act
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, sp = experience.s_t, experience.a_t,  experience.s_tp1

        if step % self.cfg.policy_update_frequency == 0:
            self.prev_omega = self.omega.copy()
            try:
                mdp = MDP(P=self.empirical_transition())
                results = self.solver.solve(self.discount_factor, mdp, self.policy.argmax(-1))
                if results.w is None:
                    return
            except Exception as e:
                print(e)
                self.updates += 1
                return
    
            omega = results.w
            if self.cfg.averaged:
                self.omega = ( self.updates * self.omega + omega) / (self.updates + 1)
            else:
                self.omega = omega
            self.updates += 1