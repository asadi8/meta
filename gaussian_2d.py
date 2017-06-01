import gym
import itertools
import matplotlib
import numpy as np
import utils
import sys
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

class meta:
    
    def __init__(self,params):
        learning_rate=params['meta_learning_rate']
        observation_size=params['policy_num_parameters']
        normal_variance=params['meta_normal_variance']
        self.state = tf.placeholder(tf.float32, [1,observation_size], "state")
        self.action = tf.placeholder(dtype=tf.float32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")
        self.W=tf.Variable(0.4*tf.ones([1]))
        self.mu =self.state*self.W
        self.mu = tf.squeeze(self.mu)
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, normal_variance)
        self.action = self.normal_dist._sample_n(1)
        self.action=tf.stop_gradient(self.action)
        self.loss = -self.normal_dist.log_prob(self.action) * self.target
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def predict(self, state, sess):
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess):
        print(target)
        print("W before update:",sess.run(self.W))
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        print("W after update:",sess.run(self.W))
        return sess.run(self.W)

'''
def train_once():
    tf.reset_default_graph()
    observation_size=200
    meta_policy = meta(learning_rate=0.0001,observation_size=observation_size,normal_variance=0.02)
    li=[]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(1000):
            state=np.random.random((1,observation_size))
            mu=sess.run(meta_policy.mu,feed_dict={meta_policy.state:state})
            action=meta_policy.predict(state,sess)
            G=-np.linalg.norm(3*state-action)
            meta_policy.update(state,G,action,sess)   
            li.append(sess.run(meta_policy.W))
    print("obsreved return:",G)
    return li

m=[]
for i in range(20):
    li=train_once()
    #plt.plot(li)
    m.append(li)
plt.plot(np.mean(m,axis=0))
plt.show()
'''