from __future__ import annotations
import numpy as np
import os
import lzma
import pickle
import torch
from numpy.typing import NDArray
from multireward_ope.continuous.agents.agent import Agent
from multireward_ope.continuous.agents.utils.utils import TimeStep, TransitionWithMaskAndNoise, \
    TorchTransitions
from multireward_ope.continuous.agents.rnd_pe.config import RNDPEConfig
from multireward_ope.continuous.networks.single_nework import SingleNetwork

class RNDPE(Agent):
    NAME = 'RNDPE'
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: RNDPEConfig,
            device: torch.device):
        super().__init__(state_dim, num_actions, num_rewards, config.epsilon_0, config.replay_capacity, device)

        # Agent components.
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._num_rewards = num_rewards
        self._cfg = config
        self._network = SingleNetwork(self._state_dim, self._num_rewards, self._num_actions, self._cfg.hidden_layer_size, None, device=device)
        self._rnd_network = SingleNetwork(self._state_dim, 1, self._num_actions, self._cfg.hidden_layer_size, None, device=device)
        self._tgt_network = SingleNetwork(self._state_dim, self._num_rewards, self._num_actions, self._cfg.hidden_layer_size,None, device=device).freeze()
        self._tgt_rnd_network = SingleNetwork(self._state_dim, 1, self._num_actions, self._cfg.hidden_layer_size,None, device=device).freeze()
        self._optimizer = torch.optim.AdamW(
            [
                {"params": self._network.parameters(), "lr": config.lr_Q},
                {"params": self._rnd_network.parameters(), "lr": config.lr_RND}
            ])
        self._total_steps = 0
        self._gradient_steps = 0


    def _gradient_step(self, batch: TorchTransitions):
        """Does a step of SGD for the whole ensemble over `transitions`."""
 
        with torch.no_grad():
            tgt_values_t = self._tgt_network.forward(batch.o_t)

            oh_t = torch.nn.functional.one_hot(batch.a_pi_t.squeeze(-1), self._num_actions).float()
            softmaxed_t = torch.nn.functional.softmax(oh_t/0.25,dim=-1).unsqueeze(1)

            q_next_target = ( tgt_values_t * softmaxed_t).sum(-1)

            target_q = batch.r_t  + self._cfg.discount * (1 - batch.d_t.unsqueeze(-1)) * q_next_target



            tgt_rnd_val = self._tgt_rnd_network.forward(batch.o_tm1).gather(-1, batch.a_tm1[:, None]).squeeze(-1)

    
        q_values = self._network.forward(batch.o_tm1).gather(-1, batch.a_tm1[:, None, None].expand(-1, self._num_rewards,1)).squeeze(-1)
        rnd_val = self._rnd_network.forward(batch.o_tm1).gather(-1, batch.a_tm1[:, None]).squeeze(-1)

        q_loss =   torch.nn.functional.mse_loss(q_values, target_q.detach())
        rnd_loss = torch.nn.functional.mse_loss(rnd_val, tgt_rnd_val.detach())
        total_loss = q_loss + rnd_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(self._rnd_network.parameters(), 2.0)
        self._optimizer.step()
        
        self._gradient_steps += 1

        self._tgt_network.soft_update(self._network, self._cfg.target_soft_update)
     

        return total_loss.item()

    @torch.no_grad()
    def qvalues(self, observation: TimeStep) -> np.ndarray:
        th_observation = torch.tensor(observation.observation[None, ...], dtype=torch.float32, device=self._device)
        values = self._network.forward(th_observation)
        q_values = values[0].cpu().numpy()

        return q_values, 0
    
    @torch.no_grad()
    def _select_action(self, obs: TimeStep) -> int:
        if np.random.rand() < self.eps_fn(self._gradient_steps):
            return np.random.choice(self._num_actions)

        th_observation = torch.tensor(obs.observation[None, ...], dtype=torch.float32, device=self._device)
        rnd_val = self._rnd_network.forward(th_observation)[0]
        rnd_tgt_val = self._tgt_rnd_network.forward(th_observation)[0]
        err = (rnd_val - rnd_tgt_val).square().softmax(-1)
        
        if np.random.rand() < self._cfg.exploration_prob:
            return np.random.choice(self._num_actions, p=err.cpu().numpy())
    
        return obs.eval_policy_action

    def select_action(self, observation: TimeStep, step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: TimeStep) -> int:
        return self._select_action(observation, greedy=True)
    
    def _update(self, timestep: TimeStep) -> None:
        """Update the agent: add transition to replay and periodically do SGD."""
        self._total_steps += 1

        timestep = timestep.to_float32()
        self._replay.add(
            TransitionWithMaskAndNoise(
                o_tm1=timestep.observation,
                a_tm1=timestep.action,
                a_pi_tm1=timestep.eval_policy_action,
                a_pi_t=timestep.eval_policy_next_action,
                r_t=timestep.rewards,
                d_t=timestep.done,
                o_t=timestep.next_observation,
                m_t=0,
                z_t=0,
            ))


        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._total_steps % self._cfg.sgd_period != 0:
            return None
        minibatch= self._replay.sample(self._cfg.batch_size)
        return self._gradient_step(TorchTransitions.from_minibatch(minibatch, self._device, self._num_rewards))

    def save_model(self, path: str, seed: int, step: int):
        model_path = f"{path}/models"
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        file_path = f"{model_path}/{self.NAME}_networks_{seed}_{step}.pkl.lzma"
        with lzma.open(file_path, 'wb') as f:
            model = {
                'network': self._network.state_dict(),
                'target_network': self._tgt_network.state_dict()
            }
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def make_default_agent(
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: RNDPEConfig,
            device: torch.device) -> RNDPE:
        """Initialize a  uniform agent with default parameters."""

        return RNDPE(
            state_dim= state_dim,
            num_actions=num_actions,
            num_rewards=num_rewards,
            config=config,
            device=device,
        )
    
