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
from multireward_ope.continuous.agents.dbmr_pe.config import DBMRPEConfig
from multireward_ope.continuous.agents.dbmr_pe.networks import ValueEnsembleWithPrior
from multireward_ope.tabular.utils import policy_evaluation

class DBMRPE(Agent):
    NAME = 'DBMR-PE'
    """Deep-Bootstrapped Multiple-Rewards PE"""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            horizon: int,
            config: DBMRPEConfig,
            device: torch.device):
        super().__init__(config.epsilon_0, config.replay_capacity, horizon, device)

        # Agent components.
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._num_rewards = num_rewards
        self._cfg = config
        self.mdp_visitation = np.ones((state_dim, num_actions, state_dim))
        self._reward_vectors = torch.eye(self._num_rewards).to(device)
        self._ensemble = ValueEnsembleWithPrior(self._state_dim,
                                                self._num_rewards,
                                                self._num_actions, 
                                                self._cfg.prior_scale,
                                                self._cfg.ensemble_size,
                                                self._cfg.hidden_layer_size,
                                                device)
        self._tgt_ensemble: ValueEnsembleWithPrior = ValueEnsembleWithPrior(self._state_dim,
                                                self._num_rewards,
                                                self._num_actions, 
                                                self._cfg.prior_scale,
                                                self._cfg.ensemble_size,
                                                self._cfg.hidden_layer_size,
                                                device).clone(self._ensemble).freeze()

        self._optimizer = torch.optim.AdamW(
            [
                {"params": self._ensemble._q_network.parameters(), "lr": config.lr_Q},
                {"params": self._ensemble._q_network_sn.parameters(), "lr": config.lr_Q},
                {"params": self._ensemble._m_network.parameters(), "lr": config.lr_M}
            ])

        # Agent state.
        self._total_steps = 0
    
        self._gradient_steps = 0
        self._start = True


    def _gradient_step(self, batch: TorchTransitions):#, idxs, weights):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        
        _batch = batch.expand_batch(self._cfg.ensemble_size, self._num_rewards)
        m_t = _batch.m_t


        # m_t = np.random.binomial(1, 0.7, (self._cfg.batch_size, self._cfg.ensemble_size)).astype(np.float32)[..., None]
        # m_t = torch.tensor(m_t, device=self._device, dtype=torch.float64)
 
    
        with torch.no_grad():
            tgt_values_t= self._tgt_ensemble.forward(_batch.o_t)
            tgt_values_tm1 = self._tgt_ensemble.forward(_batch.o_tm1)

            q_next_target = tgt_values_t.q_values.gather(-1, _batch.a_pi_t).squeeze(-1)
            target_q = _batch.r_t + _batch.z_t + self._cfg.discount * (1-_batch.d_t) * q_next_target
            
            q_next_target_sn = tgt_values_t.q_values_sn.gather(-1, _batch.a_pi_t[:,0]).squeeze(-1)
            target_q_sn = _batch.r_t[:,0] + self._cfg.discount * (1-_batch.d_t[:,0]) * q_next_target_sn
     
            
            q_values_tgt = tgt_values_tm1.q_values.gather(-1, _batch.a_tm1).squeeze(-1)
            M = (_batch.r_t + _batch.z_t + (1-_batch.d_t) * self._cfg.discount * q_next_target - q_values_tgt.detach()) / (self._cfg.discount)
            target_M = (M ** 2).detach()
    
        values = self._ensemble.forward(_batch.o_tm1)
        q_values = values.q_values.gather(-1, _batch.a_tm1).squeeze(-1)
        q_values_sn = values.q_values_sn.gather(-1, _batch.a_tm1[:,0]).squeeze(-1)
        
        # if np.random.uniform() < 1e-3:
        #     import pdb
        #     pdb.set_trace()
        #_m_t=np.random.exponential(1, size=m_t.shape)
        #m_t = torch.tensor(_m_t, dtype=torch.float32, device=m_t.device)
        # if np.random.rand() < 1e-3:
        #     import pdb
        #     pdb.set_trace()
        #weights = torch.tensor(weights, dtype=torch.float32, device=q_values.device, requires_grad=False).unsqueeze(-1).unsqueeze(-1)

        q_loss =   torch.mul(torch.square(q_values - target_q.detach()),  m_t).sum(-1).sum(-1).mean()
        
        m_values = values.m_values.gather(-1, _batch.a_tm1).squeeze(-1)

        m_loss =   torch.mul(torch.square(m_values - target_M.detach()), m_t).sum(-1).sum(-1).mean()

        tderr = q_values_sn - target_q_sn.detach()
        qsn_loss = ( torch.square(tderr).sum(-1)).mean()

        # if torch.any(_batch.r_t > 0):
        #     import pdb
        #     pdb.set_trace()

        total_loss =q_loss + m_loss +qsn_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        
        self._gradient_steps += 1

        # if self._gradient_steps % 32 == 0:
        #     self._tgt_ensemble = self._tgt_ensemble.clone(self._ensemble)
        
        self._tgt_ensemble.soft_update(self._ensemble, self._cfg.target_soft_update)
        #self.buffer.update_priorities(idxs, tderr.abs().max(-1)[0].detach().cpu().numpy() + 1e-5)


        return total_loss.item()

    @torch.no_grad()
    def qvalues(self, observation: TimeStep) -> np.ndarray:
        th_observation = torch.tensor(observation.observation[None, ...], dtype=torch.float32, device=self._device)
        values  = self._ensemble.forward(th_observation)
        
        # Size (1, ensemble size, num rewards, actions) -> (ensemble size, num_rewards, actions)
        q_values = values.q_values_sn[0].cpu().numpy()
        return q_values

    
    @torch.no_grad()
    def _select_action(self, obs: TimeStep) -> int:
        if np.random.rand() < self.eps_fn(self._gradient_steps):
            self._start = False
            return np.random.choice(self._num_actions)


        th_observation = torch.tensor(obs.observation[None, ...], dtype=torch.float32, device=self._device)
        values  = self._ensemble.forward(th_observation)
        
        # Size (1, ensemble size, num rewards, actions) -> (ensemble size, num_rewards, actions)
        q_values = values.q_values[0]#.cpu().numpy().astype(np.float64)[0]
        m_values = values.m_values[0]#.cpu().numpy().astype(np.float64)[0]
        # if np.random.rand() < 1e-3:
        #     import pdb
        #     pdb.set_trace()
        #     print(m_values.mean(0))
        
        one_hot = torch.nn.functional.one_hot(torch.tensor(obs.eval_policy_action), self._num_actions).float()
        softmaxed = torch.nn.functional.softmax(one_hot / m_values.std(0).max() )
    
        q_values = torch.quantile(q_values, self._cfg.quantile, dim=0, keepdim=False)
        m_values = torch.quantile(m_values, self._cfg.quantile, dim=0, keepdim=False) 
        
        lse_m_values = softmaxed * torch.logsumexp(m_values, dim=0)
        probs = lse_m_values / lse_m_values.sum()
        return torch.multinomial(probs, 1, replacement=True).item()
        # q_values_max = q_values.max(-1)
        # mask = np.isclose(q_values- q_values_max, 0)

        # if len(q_values[~mask]) == 0:
        #     return np.random.choice(self._num_actions)
       
        
        # if np.any(np.isnan(p)):
        #     return np.random.choice(self._num_actions)

        # return np.random.choice(self._num_actions, p=p)

    def select_action(self, observation: TimeStep, step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: TimeStep) -> int:
        return self._select_action(observation, greedy=True)
    
    def update(self, timestep: TimeStep) -> None:
        """Update the agent: add transition to replay and periodically do SGD."""
        self._total_steps += 1

        self.mdp_visitation[timestep.observation.argmax(), timestep.action, timestep.next_observation.argmax()] += 1

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
                m_t=np.random.binomial(1, 0.7,
                                    self._cfg.ensemble_size).astype(np.float32),
                #m_t=np.random.exponential(1, size=(self._cfg.ensemble_size, self._num_rewards)).astype(np.float32),
                z_t=np.random.randn(self._cfg.ensemble_size, self._num_rewards).astype(np.float32) *
                self._cfg.noise_scale,
            ))

        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._total_steps % self._cfg.sgd_period != 0:
            return None
        minibatch= self._replay.sample(self._cfg.batch_size) #, idxs, weights = 

        return self._gradient_step(TorchTransitions.from_minibatch(minibatch, self._device, self._num_rewards))#, idxs, weights)

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
    

    # def dump_buffer(self, path: str, env_config: EnvConfig, seed: int):
    #     super().dump_buffer(path, env_config, seed)
    #     file_path = f"{path}/{self.NAME}_{seed}_info.pkl.lzma"

    #     with lzma.open(file_path, 'wb') as f:
    #         pickle.dump({'history_chosen_rewards': self._history_chosen_rewards,
    #                      'history_delta_min': self._history_delta_min}, 
    #                     f, protocol=pickle.HIGHEST_PROTOCOL)