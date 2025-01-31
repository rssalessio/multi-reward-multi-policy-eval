#!/bin/bash -l
# Script to run the simulations. Please adjust the number of states, reward type accordingly
doublechain_states=15 # 8 10 15
narms_states=30 # 15 20 30
riverswim_states=30 #15 20 30
forked_riverswim_states=15 # 8 10 15
reward_set='Finite' # Finite or RewardFree
single_policy=True # True or False

python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=mr_nas_pe

python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=mr_nas_pe


python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states  agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states agent=mr_nas_pe

python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=mr_nas_pe




