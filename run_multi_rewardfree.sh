#!/bin/bash -l


#$ -P onlinepe       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00  # Specify the hard time limit for the job
#$ -N multi_rewardfree_30narms   # Give job a name
#$ -j y              # Merge the error and output streams into a single file
#$ -pe omp 32        # number of cpus
module load python3/3.10.12
module load gurobi/10.0.1
source .venv/bin/activate

doublechain_states=15 #8 10 15
narms_states=30 # 15 20 30
riverswim_states=30
forked_riverswim_states=15
reward_set='RewardFree'
single_policy=False

# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=sf_nr
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=doublechain environment.parameters.length=$doublechain_states agent=mr_nas_pe

# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=sf_nr
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=forked_riverswim environment.parameters.river_length=$forked_riverswim_states agent=mr_nas_pe


# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states  agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states agent=sf_nr
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy  environment=riverswim environment.parameters.num_states=$riverswim_states agent=mr_nas_pe

python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=noisy_policy_uniform  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=noisy_policy_visitation  agent.parameters.noise_parameter=0.30
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=sf_nr
# python tabular_sim.py experiment.reward_set=$reward_set  experiment.single_policy=$single_policy environment=narms environment.parameters.num_arms=$narms_states agent=mr_nas_pe




