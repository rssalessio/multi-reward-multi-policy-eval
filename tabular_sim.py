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
from multireward_ope.tabular.envs.make_env import make_env
from multireward_ope.tabular.reward_set import RewardSetRewardFree
from multireward_ope.tabular.agents.make_agent import make_agent
from multireward_ope.tabular.agents.base_agent import Experience

from multireward_ope.tabular.utils import policy_evaluation
from hydra.conf import ConfigStore

def run_single_experiment(seed: int, cfg: Config):
    print(f'Starting simulation {seed} - Agent: {cfg.agent.type} - Env: {cfg.environment.type}')
    np.random.seed(seed)
    env = make_env(env = cfg.environment)
    reward_set = RewardSetRewardFree(env.dim_state, env.dim_action, 
                                     RewardSetRewardFree.RewardSetFreeConfig())
    # Setup experiment
    policy_to_eval = env.default_policy(cfg.experiment.discount_factor)
    eval_rewards = reward_set.canonical_rewards()
    agent_kwargs = {'dim_state': env.dim_state,
                    'dim_action': env.dim_action,
                    'policy': policy_to_eval,
                    'rewards': reward_set,
                    **cfg.experiment}

    agent = make_agent(cfg=cfg.agent, **agent_kwargs)

    eval_rewards = reward_set.canonical_rewards()
    NUM_REWARDS = eval_rewards.shape[0]
    rewards = np.zeros((NUM_REWARDS, env.dim_state, env.dim_action))
    values = np.zeros((NUM_REWARDS, env.dim_state))
    for i in range(NUM_REWARDS):
        rewards[i, np.arange(env.dim_state), policy_to_eval] = eval_rewards[i]
        values[i] = env.policy_evaluation(rewards[i], cfg.experiment.discount_factor, policy_to_eval)

    # Start process
    s = env.reset()
    results = []

    for t in range(cfg.experiment.horizon):
        a = agent.forward(s, t)
        next_state, _ = env.step(a)
        exp = Experience(s, a, next_state)
        reset = agent.backward(exp, t)

        s = env.reset() if reset else next_state

        # Evaluate the agent
        if (t +1) % cfg.experiment.frequency_evaluation == 0:
            hat_values = np.array([
                policy_evaluation(cfg.experiment.discount_factor, 
                                  agent.empirical_transition(),
                                  R=rewards[r], policy=policy_to_eval)
                for r in range(NUM_REWARDS)])

            results.append(values - hat_values)
    
    print(agent.state_action_visits)
    return results




@hydra.main(version_base="1.2", config_path="config/tabular", config_name="config")
def run_experiments(cfg: DictConfig):
    cfg: Config = Config(**cfg)
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'Configuration {cfg}')

    with mp.Pool(cfg.experiment.num_processes) as pool:
        results = pool.starmap(run_single_experiment, [(x, cfg) for x in range(cfg.experiment.num_simulations)])
    # results = run_single_experiment(0, cfg)
    results = np.array(results)

    filename = Config.name(cfg)
    with lzma.open(f'./data/tabular/{filename}.lzma', 'wb') as f:
        pickle.dump({'cfg': cfg, 'results': results}, f)

    print("Simulation complete. Results:")
    err = np.linalg.norm(results, ord=np.inf, axis=-1)
    print(f'Avg: {err.mean(0).mean(-1)[-1]}')


if __name__ == '__main__':
    run_experiments()