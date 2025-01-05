import os
import numpy as np
import hydra
import lzma
import cvxpy as cp
import pickle
import multiprocessing as mp
from datetime import datetime
from omegaconf import DictConfig
from multireward_ope.continuous.dataclasses import Config
from multireward_ope.continuous.envs.make_env import make_env
from multireward_ope.continuous.agents.make_agent import make_agent
from multireward_ope.continuous.agents.agent import TimeStep

from multireward_ope.tabular.utils import policy_evaluation
from hydra.conf import ConfigStore

def run_single_experiment(seed: int, cfg: Config):
    print(f'Starting simulation {seed} - Agent: {cfg.agent.type} - Env: {cfg.environment.type}')
    np.random.seed(seed)
    env = make_env(env = cfg.environment)
    # Setup experiment
    agent_kwargs = {'dim_state': env.dim_state,
                    'num_actions': env.num_actions,
                    'num_rewards': env.num_rewards,
                    'horizon': env.horizon,
                    'cfg': cfg.agent,
                    'device': 'cpu'}

    agent = make_agent(**agent_kwargs)
    env_qvalues = env.compute_Q_values(env._rewards, cfg.agent.parameters.discount)

    # Start process
    curr_timestep = env.reset()
    results = []

    rewards = np.zeros(env.num_rewards)
    for t in range(cfg.experiment.horizon):
        a = agent.select_action(curr_timestep, t)
        next_timestep, _ = env.step(a)
        agent.update(next_timestep)

        rewards += next_timestep.rewards

        curr_timestep = env.reset() if next_timestep.done else next_timestep

        # Evaluate the agent
        if (t % 100) == 0 :
            qvalues = agent.qvalues(curr_timestep)[:,0]
   
            env_qval_start = env_qvalues[:,0,0,1]
            print(f'[{t}] Rewards: {qvalues} - {rewards} - Error {np.linalg.norm(qvalues-env_qval_start, axis=-1).max(0)}')
            # rewards = np.zeros(env.num_rewards)
            # if np.random.rand() < 1e-3:
            #     import pdb
            #     pdb.set_trace()
            # import pdb
            # pdb.set_trace()
        # if (t +1) % cfg.experiment.frequency_evaluation == 0:
        #     results.append(0)

    return results




@hydra.main(version_base="1.2", config_path="config/continuous", config_name="config")
def run_experiments(cfg: DictConfig):
    cfg: Config = Config(**cfg)
    date = datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
    print(f'Configuration {cfg}')

    # with mp.Pool(cfg.experiment.num_processes) as pool:
    #     results = pool.starmap(run_single_experiment, [(x, cfg) for x in range(cfg.experiment.num_simulations)])
    results = run_single_experiment(0, cfg)
    results = np.array(results)

    # filename = Config.name(cfg)
    # with lzma.open(f'./data/continuous/{filename}.lzma', 'wb') as f:
    #     pickle.dump({'cfg': cfg, 'results': results}, f)

    print("Simulation complete. Results:")
    err = np.linalg.norm(results, ord=np.inf, axis=-1)
    print(f'Avg: {err.mean(0).mean(-1)[-1]}')


if __name__ == '__main__':
    run_experiments()