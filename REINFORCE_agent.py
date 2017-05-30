import tensorflow as tf 
import numpy as np
import utils
import sys

class agent():
    gradBuffer=[]
    weights=[]
    def __init__(self, params):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,params['observation_size']],dtype=tf.float32)
        hidden=self.state_in
        inputSize=params['observation_size']
        outputSize=params['policy_num_hidden_nodes']
        for h in range(params['policy_num_hidden_layers']):
            W = tf.Variable(utils.xavier_init([inputSize, outputSize]))
            b=tf.Variable(tf.zeros(shape=[outputSize]))
            self.weights.append(W)
            self.weights.append(b)
            hidden = utils.Leaky_ReLU(tf.matmul(hidden, W) + b, leak=0.3)
            inputSize=outputSize
        W = tf.Variable(utils.xavier_init([params['policy_num_hidden_nodes'],params['num_actions']]))
        b = tf.Variable(tf.zeros(shape=[params['num_actions']]))
        self.weights.append(W)
        self.weights.append(b)
        self.output = tf.nn.softmax(tf.matmul(hidden,W)+b)

        self.return_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.return_holder)

        self.gradient_holders = []
        for idx,var in enumerate(self.weights):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss,self.weights)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['policy_learning_rate'])
        optimizer = tf.train.AdamOptimizer(learning_rate=params['policy_learning_rate'])
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,self.weights))

    def action_selection(self,s,sess):
        a_dist = sess.run(self.output,feed_dict={self.state_in:[s]})
        a = np.random.choice(range(len(a_dist[0])),p=a_dist[0])
        return a

    def initialize_for_learning(self,params):
        sess=params['tf_session']
        self.gradBuffer = sess.run(self.weights)
        for ix,grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

    def update(self,params,batch_info):
        #compute episode-based gradient estimator
        sess=params['tf_session']
        for episode_number,episode_info in enumerate(batch_info):
            returns=utils.rewardToReturn(episode_info['rewards'],params['discount_rate'])
            feed_dict={self.return_holder:returns,
                            self.action_holder:episode_info['actions'],self.state_in:np.vstack(episode_info['states'])}
            grads = sess.run(self.gradients, feed_dict=feed_dict)
            for idx,grad in enumerate(grads):
                self.gradBuffer[idx] += grad

        #now update the policy
        feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
        _ = sess.run(self.update_batch, feed_dict=feed_dict)
        #clear gradient holder
        for ix,grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0