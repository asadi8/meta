import sys
import spg
import matplotlib.pyplot as plt
import numpy as np

params={}
params['run_ID']=0
params['policy_update_freq']=20
params['policy_num_hidden_layers']=1
params['policy_num_hidden_nodes']=2
params['policy_learning_rate']=0.01
params['environment_name']='CartPole-v0'
params['num_training_episodes']=30000
params['max_time_steps']=200
params['discount_rate']=0.9999
params['PG_algorithm']='REINFORCE'
params['log_freq']=20
#***parameters of meta learner
params['meta_algorithm']='REINFORCE'
params['meta_normal_variance']=0.0001
params['meta_learning_rate']=params['meta_normal_variance']*0.0001



w,r=spg.train(params)
