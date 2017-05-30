import sys
import spg

params={}
params['run_ID']=0
params['policy_update_freq']=20
params['policy_num_hidden_layers']=1
params['policy_num_hidden_nodes']=16
params['policy_learning_rate']=0.01
params['environment_name']='CartPole-v0'
params['num_training_episodes']=3000
params['max_time_steps']=200
params['discount_rate']=0.9999
params['PG_algorithm']='REINFORCE'
params['log_freq']=100

spg.train(params)