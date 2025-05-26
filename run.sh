#!/bin/bash -l
# Script to run the simulations. 
# Please adjust the number of states, reward type and target policies accordingly

# ——— Configuration ———

state_size=small            # State space size: small, medium or large
reward_set='Finite'         # Finite or RewardFree
single_policy=True          # True or False for MultiPolicy

if [ "$state_size" = small ]; then
  doublechain_states=8
  forked_riverswim_states=8
  riverswim_states=10
  narms_states=10
elif [ "$state_size" = medium ]; then
  doublechain_states=15
  forked_riverswim_states=15
  riverswim_states=20
  narms_states=20
elif [ "$state_size" = large ]; then
  doublechain_states=30
  forked_riverswim_states=30
  riverswim_states=30
  narms_states=30
else
  echo "Unknown state_size: $state_size" >&2
  exit 1
fi

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




