import numpy, sys
import tensorflow as tf

def rewardToReturn(rewards,gamma):# takes a list of reward per t and converts it to return per t
    T=len(rewards)
    returns=T*[0]
    returns[T-1]=rewards[T-1] 
    for t in range(T-2,-1,-1):
        returns[t]=rewards[t]+gamma*returns[t+1]
    return returns

def rep_2_rep_and_action(rep,action,actionSize):
        rep_and_action=numpy.zeros((1,rep.shape[1]+actionSize))
        rep_and_action[0,0:rep.shape[1]]=rep
        one_hot_action=numpy.zeros((1,actionSize))
        one_hot_action[0,action]=1
        rep_and_action[0,rep.shape[1]:rep.shape[1]+actionSize]=one_hot_action
        return rep_and_action

def printLog(episode,info,batch_episode_number,frequency):
    if (episode) % frequency ==0:
        print("***")
        print("episode number",episode)
        print([a[0] for a,b,c,d in info[-50:]])
        average_return=numpy.mean([a[0] for a,b,c,d in info[-batch_episode_number:]])
        print("average return",average_return)
        print("***")
        sys.stdout.flush()

def save_stuff(actor,info,episode,run,batch_episode_number):
    if (episode) % batch_episode_number ==0:
        numpy.savetxt(str(run)+"-"+str(episode)+".txt",[numpy.mean([a[0] for a,b,c,d in info[-batch_episode_number:]])])
        actor.model.save_weights(str(run)+"-"+str(episode)+".h5")
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def Leaky_ReLU(x, leak=0.3, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def interact_one_episode(env,sess,params,myAgent):
    s,t,done,G=env.reset(),0,False,0
    states,actions,rewards=[],[],[]
    while t<params['max_time_steps'] and done==False:
        states.append(s)
        a=myAgent.action_selection(s,sess)
        actions.append(a)
        s_p,r,done,_ = env.step(a)
        rewards.append(r)
        G += r
        s,t=s_p,t+1#go to next time step

    episode_info={}
    episode_info['states']=states
    episode_info['actions']=actions
    episode_info['rewards']=rewards
    episode_info['return']=G
    return episode_info
    
def print_performance(params,episode_number,return_per_episode):
    if episode_number % params['log_freq'] == 0 and episode_number>0:#print how well the agent is doing
        print("episode "+str(episode_number))
        print("return mean: ",str(numpy.mean(return_per_episode[-params['log_freq']:])))
        print("retrun std: ",str(numpy.std(return_per_episode[-params['log_freq']:])))
        print("retrun max: ",str(numpy.max(return_per_episode[-params['log_freq']:])))
        print("retrun min: ",str(numpy.min(return_per_episode[-params['log_freq']:])))
        #print(return_per_episode[params['log_freq']:])
        #print(params['log_freq'])
        sys.stdout.flush() 