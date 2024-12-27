from __future__ import annotations
import numpy as np
import os
import lzma
import pickle
import torch
from numpy.typing import NDArray
from multireward_ope.continuous.agents.agent import Agent, EnvConfig
from multireward_ope.continuous.agents.utils.utils import TimeStep, TransitionWithMaskAndNoise, \
    TorchTransitions
from multireward_ope.continuous.agents.dbmr_pe.config import DBMRPEConfig
from multireward_ope.continuous.agents.dbmr_pe.networks import ValueEnsembleWithPrior


golden_ratio = (1+np.sqrt(5))/2
golden_ratio_sq = golden_ratio ** 2


class DBMRPE(Agent):
    NAME = 'DBMR-PE'
    """Deep-Bootstrapped Multiple-Rewards PE"""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: DBMRPEConfig,
            device: torch.device):
        super().__init__(config.epsilon_0, config.replay_capacity, device)

        # Agent components.
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._num_rewards = num_rewards
        self._cfg = config
        self._reward_vectors = torch.eye(self._num_rewards).to(device)
        self._ensemble = ValueEnsembleWithPrior(self._state_dim,
                                                self._num_rewards,
                                                self._num_actions, 
                                                self._cfg.prior_scale,
                                                self._cfg.ensemble_size,
                                                self._cfg.hidden_layer_size,
                                                device,
                                                self._torch_rng)
        self._tgt_ensemble: ValueEnsembleWithPrior = ValueEnsembleWithPrior(self._state_dim,
                                                self._num_rewards,
                                                self._num_actions, 
                                                self._cfg.prior_scale,
                                                self._cfg.ensemble_size,
                                                self._cfg.hidden_layer_size,
                                                device,
                                                self._torch_rng).clone(self._ensemble).freeze()

        self._optimizer = torch.optim.Adam(
            [
                {"params": self._ensemble._q_network.parameters(), "lr": config.lr_Q},
                {"params": self._ensemble._m_network.parameters(), "lr": config.lr_M}
            ])

        # Agent state.
        self._total_steps = 0
        self._active_reward = 0
        self._chosen_rewards = np.zeros(self._num_rewards)
        self._chosen_rewards[self._active_reward] = 1
    
        self._delta_min = torch.nn.Parameter(torch.zeros(self._num_rewards, device=device), requires_grad=True)
        
        self._avg_delta_min = np.zeros(self._num_rewards)
        self._delta_min_optim = torch.optim.NAdam([self._delta_min], lr=1e-3)
        self.uniform_number = self._np_rng.uniform()
        self._gradient_steps = 0
        self._start = True

        self._history_chosen_rewards = []
        self._history_delta_min = []

    def _gradient_step(self, batch: TorchTransitions):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        
        _batch = batch.expand_batch(self._cfg.ensemble_size, self._num_rewards)
        m_t = _batch.m_t
        if self._cfg.enable_mix:
            m_t = self._np_rng.binomial(1, self._cfg.mask_prob, (self._cfg.batch_size, self._cfg.ensemble_size)).astype(np.float32)[..., None]
            m_t = torch.tensor(m_t, device=self._device, dtype=torch.float64)
    
        with torch.no_grad():
            q_values_target = self._tgt_ensemble.forward(_batch.o_t).q_values
            next_actions = self._ensemble.forward(_batch.o_t).q_values.max(-1)[1]
            q_target = q_values_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_q = _batch.r_t + _batch.z_t + self._cfg.discount * (1-_batch.d_t) * q_target
            
            values_tgt = self._tgt_ensemble.forward(_batch.o_tm1).q_values
            q_values_tgt = values_tgt.gather(-1, _batch.a_tm1).squeeze(-1)
            M = (_batch.r_t + _batch.z_t + (1-_batch.d_t) * self._cfg.discount * q_target - q_values_tgt.detach()) / (self._cfg.discount)
            target_M = (M ** (2 ** self._cfg.kbar)).detach()
    
        values = self._ensemble.forward(_batch.o_tm1)
        q_values = values.q_values.gather(-1, _batch.a_tm1).squeeze(-1)
      
        q_loss =   torch.mul(torch.square(q_values - target_q.detach()),  m_t).mean()
        
        m_values = values.m_values.gather(-1, _batch.a_tm1).squeeze(-1)

        m_loss =   torch.mul(torch.square(m_values - target_M.detach()), m_t).mean()
        total_loss =q_loss + m_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        
        self._gradient_steps += 1
        
        self._tgt_ensemble.soft_update(self._ensemble, self._cfg.target_soft_update)

    
        with torch.no_grad():
            q_target = self._ensemble.forward(_batch.o_tm1).q_values
            esitm = (-q_target.topk(2)[0].diff(dim=-1).squeeze(-1))#.cpu().numpy()
            estim_delta_min = esitm.min(1)[0].mean(0).detach()#np.quantile(esitm, q=0.25, axis=1).mean(0)
            
        for g in self._delta_min_optim.param_groups:
            g['lr'] = 10 ** np.random.uniform(*self._cfg.lr_delta_min)

        loss_min = torch.nn.functional.huber_loss(self._delta_min, estim_delta_min)
        self._delta_min_optim.zero_grad()
        loss_min.backward()
        self._delta_min_optim.step()
    
        n = self._gradient_steps
        x = self._delta_min.detach().cpu().numpy()
        self._avg_delta_min =((n - 1) * self._avg_delta_min / n) + (x / n)
        self._history_delta_min.append(x.copy())

        return total_loss.item()
    
    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32]) -> int:
        if self._np_rng.rand() < self.eps_fn(self._gradient_steps):
            self._start = False
            return self._np_rng.randint(self._num_actions)

        if self._cfg.per_step_randomization:
            self.uniform_number = self._np_rng.uniform()

        observation = torch.tensor(observation[None, ...], dtype=torch.float32, device=self._device)
        values  = self._ensemble.forward(observation)
        
        # Size (1, ensemble size, num rewards, actions) -> (ensemble size, num_rewards, actions)
        q_values = values.q_values.cpu().numpy().astype(np.float64)[0]
        m_values = values.m_values.cpu().numpy().astype(np.float64)[0]
        
        if self._start:
            self.uniform_number = self._np_rng.uniform()
            t = self._chosen_rewards.sum()
            delta = np.sqrt(t) - self._num_rewards / 2
            if np.any(self._chosen_rewards - delta < 0):
                self._active_reward = np.argmin(self._chosen_rewards)
            elif self._np_rng.rand() < self._cfg.exploration_prob:
                self._active_reward = self._np_rng.randint(self._num_rewards)
            else:
                weight = 1/ (1e-6+self._avg_delta_min) ** 2
                weight = weight/weight.sum()
                self._active_reward = self._np_rng.choice(self._num_rewards, p=weight)
            self._chosen_rewards[self._active_reward] += 1
            self._history_chosen_rewards.append(self._active_reward)
            self._start = False
    
        q_values = np.quantile(q_values[:, self._active_reward], self.uniform_number, axis=0, keepdims=False)
        m_values = np.quantile(m_values[:, self._active_reward], self.uniform_number, axis=0, keepdims=False)** (2 ** (1- self._cfg.kbar))
        q_values_max = q_values.max(-1)
        mask = np.isclose(q_values- q_values_max, 0)

        if len(q_values[~mask]) == 0:
            return self._np_rng.choice(self._num_actions)
        delta = q_values.max() - q_values
        delta[mask] = self._avg_delta_min[self._active_reward] * ((1 - self._cfg.discount)) / (1 + self._cfg.discount)


        Hsa = (2 + 8 * golden_ratio_sq * m_values) / np.clip((delta ** 2), 1e-16, np.inf)
        if np.any(np.isnan(Hsa)):
            return self._np_rng.choice(self._num_actions)

        C = np.max(np.maximum(4, 16 * (self._cfg.discount ** 2) * golden_ratio_sq * m_values[mask]))
        Hopt = C / (delta[mask] ** 2)

        Hsa[mask] = np.sqrt(  Hopt * Hsa[~mask].sum(-1)* 2 / (self._state_dim * (1 - self._cfg.discount)))
        H = Hsa * 1e-14
        p = (H/H.sum(-1, keepdims=True))
        
        if np.any(np.isnan(p)):
            return self._np_rng.choice(self._num_actions)

        return self._np_rng.choice(self._num_actions, p=p)

    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        return self._select_action(observation, greedy=True)
    
    def update(self, timestep: TimeStep) -> None:
        """Update the agent: add transition to replay and periodically do SGD."""
        self._total_steps += 1

        if timestep.done:
            self.uniform_number = self._np_rng.uniform()
            self._start = True

        timestep = timestep.to_float32()
        self._replay.add(
            TransitionWithMaskAndNoise(
                o_tm1=timestep.observation,
                a_tm1=timestep.action,
                r_t=timestep.rewards,
                d_t=timestep.done,
                o_t=timestep.next_observation,
                m_t=self._np_rng.binomial(1, self._cfg.mask_prob,
                                    self._cfg.ensemble_size).astype(np.float32),
                z_t=self._np_rng.randn(self._cfg.ensemble_size).astype(np.float32) *
                self._cfg.noise_scale,
            ))

        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._total_steps % self._cfg.sgd_period != 0:
            return None
        minibatch = self._replay.sample(self._cfg.batch_size)
        return self._gradient_step(TorchTransitions.from_minibatch(minibatch, self._device, self._num_rewards))

    def save_model(self, path: str, seed: int, step: int):
        model_path = f"{path}/models"
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        file_path = f"{model_path}/{self.NAME}_networks_{seed}_{step}.pkl.lzma"
        with lzma.open(file_path, 'wb') as f:
            model = {
                'network': self._ensemble.state_dict(),
                'target_network': self._tgt_ensemble.state_dict()
            }
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def make_default_agent(
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: DBMRPEConfig,
            device: torch.device) -> DBMRPE:
        """Initialize a Bootstrapped DQN agent with default parameters."""

        return DBMRPE(
            state_dim= state_dim,
            num_actions=num_actions,
            num_rewards=num_rewards,
            config=config,
            device=device,
        )
    

    def dump_buffer(self, path: str, env_config: EnvConfig, seed: int):
        super().dump_buffer(path, env_config, seed)
        file_path = f"{path}/{self.NAME}_{seed}_info.pkl.lzma"

        with lzma.open(file_path, 'wb') as f:
            pickle.dump({'history_chosen_rewards': self._history_chosen_rewards,
                         'history_delta_min': self._history_delta_min}, 
                        f, protocol=pickle.HIGHEST_PROTOCOL)