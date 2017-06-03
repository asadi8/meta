import tensorflow as tf
import numpy as np
import gym
import sys
import utils
import REINFORCE_agent
import gaussian_2d
import matplotlib.pyplot as plt
Rmax=200
def train(params):
    tf.reset_default_graph() #Clear the Tensorflow graph.
    env = gym.make(params['environment_name'])#create an environment
    params['observation_size']=env.observation_space.shape[0]#get observation size
    params['num_actions']=env.action_space.n#and number of actions from the domain

    if params['PG_algorithm']=='REINFORCE':#choose the learning algorithm
        myAgent = REINFORCE_agent.agent(params)
        params['policy_num_parameters']=myAgent.num_policy_parameters()
    if params['meta_algorithm']=='REINFORCE':
        meta_learner=gaussian_2d.meta(params)

    init = tf.initialize_all_variables()#launch tf variables

    with tf.Session() as sess:

        params['tf_session']=sess
        sess.run(init)
        return_per_episode = []#this is solely for log purpose
        batch_mean_return =[]
        myAgent.initialize_for_learning(params)#initialize the learning algorithm
        batch_info=[]#data to learn from goes here
        Ws=[]

        for episode_number in range(params['num_training_episodes']):

            episode_info=utils.interact_one_episode(env,sess,params,myAgent)#do one rollout
            batch_info.append(episode_info)#and store everything relevant
            return_per_episode.append(episode_info['return'])

            if episode_number % params['policy_update_freq'] == 0 and episode_number > 0:#update policy network
                batch_mean_return.append(np.mean(return_per_episode[-params['policy_update_freq']:]))
                if len(batch_mean_return)>1:
                    #meta_return=(batch_mean_return[-1]-batch_mean_return[-2])/Rmax
                    meta_return=-np.linalg.norm(meta_state-meta_action)
                    #print("batch mean return is the following",batch_mean_return)
                    print("W is")
                    print("meta_return is the following",meta_return)
                    #print(meta_state-meta_action)
                    #sys.exit(1)
                    W=meta_learner.update(meta_state, meta_return, meta_action, sess)
                    #sys.exit(1)

                meta_state,meta_action=myAgent.update(params,batch_info,meta_learner)
                
                
                #Ws.append(W)
                #sys.exit(1)

                batch_info=[]#clear data holder
            #utils.print_performance(params,episode_number,return_per_episode)#print how well the policy is doing and
            #go to next episode
    return Ws,return_per_episode
