import os
import numpy as np
import hydra
import lzma
import cvxpy as cp
import pickle
import multiprocessing as mp
from datetime import datetime
from omegaconf import DictConfig
from multireward_ope.tabular.dataclasses import Config
from multireward_ope.tabular.envs.make_env import make_env, Env
from multireward_ope.tabular.reward_set import RewardSetRewardFree, RewardSetType, RewardSetFinite, RewardSet
from multireward_ope.tabular.agents.make_agent import make_agent
from multireward_ope.tabular.agents.base_agent import Experience
from multireward_ope.tabular.policy import Policy
from typing import Sequence, Tuple
from multireward_ope.tabular.utils import policy_evaluation, policy_iteration
from hydra.conf import ConfigStore

def setup_rewards(env: Env, reward_set_type: RewardSetType, single_policy: bool, discount_factor: float) -> Sequence[Tuple[RewardSet, Policy, np.ndarray]]:
    num_policies = 1 if single_policy else 3
    if reward_set_type == RewardSetType.REWARD_FREE:
        reward_set = [RewardSetRewardFree(env.dim_state, env.dim_action, 
                                        RewardSetRewardFree.RewardSetFreeConfig()) for _ in range(num_policies)]
        idxs_rewfree = np.random.choice(reward_set[0].eval_rewards().shape[0], 3, replace=False)


    elif reward_set_type == RewardSetType.FINITE:
        rfset = RewardSetRewardFree(env.dim_state, env.dim_action, 
                                        RewardSetRewardFree.RewardSetFreeConfig())
        canonical_rewards = rfset.canonical_rewards()
        
        idxs = np.random.choice(canonical_rewards.shape[0], 3, replace=False)
        reward_set = [RewardSetFinite(env.dim_state, env.dim_action,
                                     RewardSetFinite.RewardSetFiniteConfig(canonical_rewards[id][None,...])) for id in idxs]
        
    else:
        raise Exception(f'Reward set {reward_set_type} not found!')
    # Compute values and policies

    if single_policy:
        eval_rewards = reward_set[0].eval_rewards()
        rewards = np.zeros((eval_rewards.shape[0], env.dim_state, env.dim_action))
        values = np.zeros((eval_rewards.shape[0], env.dim_state))
        eval_policy = env.default_policy(discount_factor)
        for i in range(eval_rewards.shape[0]):
            rewards[i, np.arange(env.dim_state), eval_policy] = eval_rewards[i]
            values[i] = env.policy_evaluation(rewards[i], discount_factor, eval_policy)
        return [(reward_set[0], eval_policy, values)]
    else:
        returns=[]
        for id in range(num_policies):
            # Compute optimal policies from some selected rewards
            if reward_set[id].set_type == RewardSetType.REWARD_FREE:
                reward_to_eval=reward_set[id].eval_rewards()[idxs_rewfree[id]]
            else:
                reward_to_eval=reward_set[id].eval_rewards()

            rew = np.zeros((env.dim_state, env.dim_action))
            rew[np.arange(env.dim_state), env.default_policy(discount_factor)] = reward_to_eval
            _,pol,_ =env.policy_iteration(rew, discount_factor)
            
            # Compute values for those policies on the evaluation rewards
            eval_rewards = reward_set[id].eval_rewards()
            rewards = np.zeros((eval_rewards.shape[0], env.dim_state, env.dim_action))
            values = np.zeros((eval_rewards.shape[0], env.dim_state))
            for i in range(eval_rewards.shape[0]):
                rewards[i, np.arange(env.dim_state), pol] = eval_rewards[i]
                values[i] = env.policy_evaluation(rewards[i], discount_factor, pol)
            
            returns.append((reward_set[id], pol, values))


    return returns

def run_single_experiment(seed: int, cfg: Config):
    print(f'Starting simulation {seed} - Agent: {cfg.agent.type} - Env: {cfg.environment.type}')
    np.random.seed(seed)
    env = make_env(env = cfg.environment)

    rewards_policies = setup_rewards(env, cfg.experiment.reward_set, cfg.experiment.single_policy, cfg.experiment.discount_factor)

 
    num_policies = len(rewards_policies)
    agent_kwargs = {'dim_state': env.dim_state,
                    'dim_action': env.dim_action,
                    'rewards_policies': rewards_policies,
                    **cfg.experiment}

    agent = make_agent(cfg=cfg.agent, **agent_kwargs)
    
    
    # Start process
    s = env.reset()
    results = {'rel_error': [], 'abs_error': []}

    for t in range(cfg.experiment.horizon):
        a = agent.forward(s, t)
        next_state, _ = env.step(a)
        exp = Experience(s, a, next_state)
        reset = agent.backward(exp, t)

        s = env.reset() if reset else next_state

        # Evaluate the agent
        if (t +1) % cfg.experiment.frequency_evaluation == 0:
            hat_values = []
            true_values = []
            for reward_set, policy_to_eval, values in rewards_policies:
                eval_rewards = reward_set.eval_rewards()
                for id in range(eval_rewards.shape[0]):
                    rew = np.zeros((env.dim_state, env.dim_action))
                    rew[np.arange(env.dim_state), policy_to_eval] = eval_rewards[id]
                    hat_values_r_pol = np.array(
                        policy_evaluation(cfg.experiment.discount_factor, 
                                        agent.empirical_transition(),
                                        R=rew, policy=policy_to_eval))
                    hat_values.append(hat_values_r_pol)
                    true_values.append(values[id])
            hat_values = np.array(hat_values)
            true_values = np.array(true_values)

            results['rel_error'].append(1 - hat_values/true_values)
            results['abs_error'].append(true_values - hat_values)
    
            
            rel_err = np.linalg.norm(results['rel_error'][-1], ord=np.inf, axis=-1).max()
            abs_err = np.linalg.norm(results['abs_error'][-1], ord=np.inf, axis=-1).max()
            #print(f'[{t}] Abs Err = {abs_err} / Rel Err ={rel_err}')
  
    return results




@hydra.main(version_base="1.2", config_path="config/tabular", config_name="config")
def run_experiments(cfg: DictConfig):
    cfg: Config = Config(**cfg)
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'Configuration {cfg}')

    with mp.Pool(cfg.experiment.num_processes) as pool:
        results = pool.starmap(run_single_experiment, [(x, cfg) for x in range(cfg.experiment.num_simulations)])
    # results = [run_single_experiment(0, cfg)]
    rel_results = np.array([res['rel_error'] for res in results])
    abs_results = np.array([res['abs_error'] for res in results])


    filename = Config.name(cfg)
    with lzma.open(f'./data/tabular/{filename}.lzma', 'wb') as f:
        pickle.dump({'cfg': cfg, 'rel_results': rel_results, 'abs_results': abs_results}, f)

    print("Simulation complete. Results:")
    err = np.linalg.norm(abs_results[:,-1], ord=np.inf, axis=-1).max(-1)

    print(f'Avg: {err.mean()}/{err.std()}')


if __name__ == '__main__':
    run_experiments()