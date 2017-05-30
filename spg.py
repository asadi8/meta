import tensorflow as tf
import numpy as np
import gym
import sys
import utils
import REINFORCE_agent

def train(params):
    tf.reset_default_graph() #Clear the Tensorflow graph.
    env = gym.make(params['environment_name'])#create an environment
    params['observation_size']=env.observation_space.shape[0]#get observation size
    params['num_actions']=env.action_space.n#and number of actions from the domain

    if params['PG_algorithm']=='REINFORCE':#choose the learning algorithm
        myAgent = REINFORCE_agent.agent(params)

    init = tf.initialize_all_variables()#launch tf variables

    with tf.Session() as sess:

        params['tf_session']=sess
        sess.run(init)
        return_per_episode = []#this is solely for log purpose
        myAgent.initialize_for_learning(params)#initialize the learning algorithm
        batch_info=[]#data to learn from goes here

        for episode_number in range(params['num_training_episodes']):

            episode_info=utils.interact_one_episode(env,sess,params,myAgent)#do one rollout
            batch_info.append(episode_info)#and store everything relevant
            return_per_episode.append(episode_info['return'])

            if episode_number % params['policy_update_freq'] == 0 and episode_number > 0:#update policy network
                myAgent.update(params,batch_info)
                batch_info=[]#clear data holder

            if episode_number % params['log_freq'] == 0 and episode_number>0:#print how well the agent is doing
                print("episode "+str(episode_number)+
                    " - return: ",str(np.mean(return_per_episode[params['log_freq']:])))
                sys.stdout.flush()