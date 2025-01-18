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
            config: DBMRPEConfig,
            device: torch.device):
        super().__init__(state_dim, num_actions, num_rewards, config.epsilon_0, config.replay_capacity, device)

        # Agent components.
        self._cfg = config
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
                {"params": self._ensemble._m_network.parameters(), "lr": config.lr_M}
            ])

        # Agent state.
        self._total_steps = 0
    
        self._gradient_steps = 0
        self._start = True
        self.idx = np.random.choice(self._num_rewards)
        self.idx_head = np.random.choice(self._cfg.ensemble_size)
        self.chosen_rewards = np.zeros(self._num_rewards)
        self.chosen_rewards[self.idx] += 1
        self.const = max(0.1,np.random.exponential(scale=2))
        self.prev_rms_std = 0
        self.curr_rms_std = 1
        self.max_std = 1e-3


    def _gradient_step(self, batch: TorchTransitions):#, idxs, weights):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        
        _batch = batch.expand_batch(self._cfg.ensemble_size, self._num_rewards)
        m_t = _batch.m_t

        #m_t = m_t.exponential_()
        # m_t = np.random.binomial(1, 0.7, (self._cfg.batch_size, self._cfg.ensemble_size)).astype(np.float32)[..., None]
        # m_t = torch.tensor(m_t, device=self._device, dtype=torch.float64)
 
    
        with torch.no_grad():
            values_t= self._ensemble.forward(_batch.o_t)
            tgt_values_t= self._tgt_ensemble.forward(_batch.o_t)
            tgt_values_tm1 = self._tgt_ensemble.forward(_batch.o_tm1)

            oh_t = torch.nn.functional.one_hot(_batch.a_pi_t.squeeze(-1), self._num_actions).float()
            softmaxed_t = torch.nn.functional.softmax(oh_t/0.25,dim=-1)
    

            q_next_target = ( tgt_values_t.q_values * softmaxed_t).sum(-1)
            target_q = _batch.r_t  + self._cfg.discount * (1-_batch.d_t) * q_next_target

            # M
            q_values_tgt = tgt_values_tm1.q_values.gather(-1, _batch.a_tm1).squeeze(-1)
            delta = (_batch.r_t + self._cfg.discount * q_next_target - q_values_tgt.detach()) / (self._cfg.discount)
            #M = (1-_batch.d_t) * self._cfg.discount * q_next_target - q_values_tgt.detach()
            ind = (_batch.a_tm1 == _batch.a_pi_tm1).float().squeeze(-1)
            next_actions = values_t.m_values.argmax(-1, keepdim=True)
            next_M = tgt_values_t.m_values.gather(-1, next_actions).squeeze(-1)
         
            
            target_M = (delta.square()  + q_values_tgt.std(1, keepdim=True)).mul(ind) + (1-_batch.d_t) *self._cfg.discount * next_M
    
        values = self._ensemble.forward(_batch.o_tm1)
        q_values = values.q_values.gather(-1, _batch.a_tm1).squeeze(-1)
        
 
        q_loss =   torch.mul(torch.square(q_values - target_q.detach()),  m_t).mean()
        
        m_values = values.m_values.gather(-1, _batch.a_tm1).squeeze(-1)

        m_loss =   torch.mul(torch.square(m_values - target_M.detach()), m_t).mean()
        total_loss =q_loss + m_loss 
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._ensemble.parameters(), 2.0)

        self._optimizer.step()
        
        self._gradient_steps += 1

        # if self._gradient_steps % 32 == 0:
        #     self._tgt_ensemble = self._tgt_ensemble.clone(self._ensemble)
        
        self._tgt_ensemble.soft_update(self._ensemble, self._cfg.target_soft_update)
        #self.buffer.update_priorities(idxs, tderr.abs().max(-1)[0].detach().cpu().numpy() + 1e-5)
        self.prev_rms_std = self.curr_rms_std
        self.curr_rms_std =  0.95*self.curr_rms_std + 0.05* values.q_values.std(1).max().item()
        if self.curr_rms_std > self.max_std:
            self.max_std = self.curr_rms_std
        


        return total_loss.item()

    @torch.no_grad()
    def qvalues(self, observation: TimeStep) -> np.ndarray:
        th_observation = torch.tensor(observation.observation[None, ...], dtype=torch.float32, device=self._device)
        values  = self._ensemble.forward(th_observation)
        # std = max(values.q_values[0].std(0).max().item(), values.m_values[0].std(0).max().item())
        # std = max(0.5,np.exp(3*std) - 0.5)
        # Size (1, ensemble size, num rewards, actions) -> (ensemble size, num_rewards, actions)
        q_values = values.q_values[0].mean(0).cpu().numpy()
        const= max(values.q_values[0].std(0).max().item(), values.m_values[0].std(0).max().item())
        C0 = np.log(self._num_actions * 0.95 /(1- 0.95))
        b = 2*(np.exp(C0 - 0.5) -1) 
        C = C0 - np.log(1+b*values.q_values[0].std(0).max().item())
        C = C0 - np.log(1+b* max(const/self.curr_rms_std,0))


        print(f'CHOSEN REW  {self.chosen_rewards}')

        return q_values, (const/self.curr_rms_std , const/self.max_std, values.m_values[0].std(0).max().item(), values.q_values[0].std(0).max().item())

    
    @torch.no_grad()
    def _select_action(self, obs: TimeStep) -> int:
        if np.random.rand() < self.eps_fn(self._gradient_steps):
            self._start = False
            return np.random.choice(self._num_actions)


        th_observation = torch.tensor(obs.observation[None, ...], dtype=torch.float32, device=self._device)
        values  = self._ensemble.forward(th_observation)
        
        # Size (1, ensemble size, num rewards, actions) -> (ensemble size, num_rewards, actions)
        q_values = values.q_values[0,self.idx_head, self.idx]#.cpu().numpy().astype(np.float64)[0]
        m_values = values.m_values[0,self.idx_head, self.idx]#.cpu().numpy().astype(np.float64)[0]

        # rr = np.random.rand()
        # q_values = torch.quantile(q_values, rr, dim=0, keepdim=False)
        # m_values = torch.quantile(m_values, rr, dim=0, keepdim=False) 

        return m_values.argmax().item()
        # std = max(q_values.std(0).max().item(), m_values.std(0).max().item())
        # # std = max(0.5,np.exp(3*std) - 0.5)
        
        # one_hot = torch.nn.functional.one_hot(torch.tensor(obs.eval_policy_action), self._num_actions).float()
        
        # const = torch.max(values.q_values[0].std(0).max(), values.m_values[0].std(0).max()).item()

        # C0 = np.log(self._num_actions * 0.95 /(1- 0.95))
        # b = 2*(np.exp(C0 - 0.5) -1) 
        # C = C0 - np.log(1+b* max(const/self.max_std,0))

        # softmaxed = torch.nn.functional.softmax(C * one_hot)#/ 2)#(1 + std))
    
        # rr = np.random.rand()
        # q_values = torch.quantile(q_values, rr, dim=0, keepdim=False)
        # m_values = torch.quantile(m_values, rr, dim=0, keepdim=False) 
        
        # #lse_m_values = softmaxed * torch.logsumexp(m_values, dim=0)
        # lse_m_values = torch.logsumexp(m_values, dim=0)
        # if np.random.rand() < 1e-3:
        #     print(lse_m_values)
        # lse_m_values=  lse_m_values.div(2).softmax(dim=-1)
        # #lse_m_values = lse_m_values / lse_m_values.sum()
        # eps = min(1,const/self.curr_rms_std)
        # probs = eps * lse_m_values + (1-eps) * one_hot
        # # probs =  probs #+ 0.1 * torch.ones(self._num_actions, device=probs.device)/self._num_actions
        # return torch.multinomial(probs, 1, replacement=True).item()
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
    
    def _update(self, timestep: TimeStep) -> None:
        """Update the agent: add transition to replay and periodically do SGD."""
        self._total_steps += 1

        if self._start:
            self._start = False
            


        if timestep.done:
            self._start = True
            self.idx_head = np.random.choice(self._cfg.ensemble_size)
            U = self.chosen_rewards < np.sqrt(self.chosen_rewards.sum()) - self._num_rewards
            if np.any(U):
                self.idx = self.chosen_rewards.argmin()
            else:
                with torch.no_grad():
                    th_observation = torch.tensor(timestep.observation[None, ...], dtype=torch.float32, device=self._device)
                    values  = (self._ensemble.forward(th_observation).m_values[0].max(0)[0].mean(-1)).softmax(dim=0)
                    print(f'VALUES {values}')
                    # import pdb
                    # pdb.set_trace()
                    if np.random.rand() < 0.2:
                        self.idx = np.random.choice(self._num_rewards)
                    else:
                        self.idx = np.random.choice(self._num_rewards, p = values.cpu().numpy())
            self.chosen_rewards[self.idx] += 1
            
        
    
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
                m_t=np.random.binomial(1, 0.5,
                                    (self._cfg.ensemble_size, self._num_rewards)).astype(np.float32),
                z_t=np.random.randn(self._cfg.ensemble_size, 1).astype(np.float32) * self._cfg.noise_scale,
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