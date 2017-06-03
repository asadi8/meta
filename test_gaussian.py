import gym
import itertools
import matplotlib
import numpy as np
import utils
import sys
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

class meta_policy():
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None,1,2], "state")
            self.action = tf.placeholder(dtype=tf.float32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            W=tf.Variable(utils.xavier_init([2, 1]))
            b=tf.Variable(tf.zeros(shape=[1]))
            self.mu =tf.matmul(self.state,W) + b
            self.mu = tf.squeeze(self.mu)
            
            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            
            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5

            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action=tf.stop_gradient(self.action)

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            #self.loss -= 1e-1 * self.normal_dist.entropy()
            
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
    
    def predict(self, state, sess):
        return sess.run(self.action, { self.state: [state] })

    def update(self, state, target, action, sess):
        feed_dict = { self.state: [state], self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

tf.reset_default_graph()


env = gym.envs.make("MountainCarContinuous-v0")
env.observation_space.sample()
meta_policy = meta_policy(learning_rate=0.005)
mus=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    state=env.reset()
    for t in range(1000):
        print(np.array(state).shape)
        mu=sess.run(meta_policy.mu,feed_dict={meta_policy.state:[state]})
        print(mu)
        action=meta_policy.predict(state,sess)
        #print("action:",action)
        G=-np.square(action-5)
        #print("obsreved return:",G)
        meta_policy.update(state,G,action,sess)   
        mus.append(mu)
    plt.plot(mus)
    plt.show()
plt.close()
