import sys
import spg
import matplotlib.pyplot as plt
import numpy as np

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
params['log_freq']=20
#***parameters of meta learner
params['meta_algorithm']='REINFORCE'
params['meta_learning_rate']=0.000001
params['meta_normal_variance']=0.1
if sys.argv[1]=='new':
    w_list=[]
    r_list=[]
    for run in range(int(sys.argv[2])):
        w,r=spg.train(params)
        w_list.append(w)
        r_list.append(r)
        np.savetxt("w"+str(run)+".txt",w)
        np.savetxt("r"+str(run)+".txt",r)
elif sys.argv[1]=='stored':
    w_list=[]
    r_list=[]
    for run in range(int(sys.argv[2])):
        w=np.loadtxt("w"+str(run)+".txt")
        r=np.loadtxt("r"+str(run)+".txt")
        w_list.append(w)
        r_list.append(r)


plt.subplot(211)
plt.plot(np.mean(w_list,axis=0))
plt.subplot(212)
plt.plot(np.mean(r_list,axis=0))
plt.show()
plt.close()