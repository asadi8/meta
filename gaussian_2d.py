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
        self.state = tf.placeholder(tf.float32, [1,observation_size], "state")
        self.action = tf.placeholder(dtype=tf.float32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")
        self.W=tf.Variable(-1*tf.ones([1]))
        self.mu =tf.multiply(self.state,self.W)
        #self.mu = tf.squeeze(self.mu)

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
        self.loss = -self.normal_dist.log_prob(self.action) * self.target
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.raw_grads=self.optimizer.compute_gradients(self.loss)
        #self.raw_grads=[(tf.clip_by_value(grad,-20,20),var) for grad,var in self.raw_grads]
        self.train_op = self.optimizer.apply_gradients(self.raw_grads)
    
    def predict(self, state, sess):
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess):
        #print(target)
        #print("W before update:",sess.run(self.W))
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss,grads = sess.run([self.train_op, self.loss,self.raw_grads], feed_dict)

        #print("grads",grads)
        '''
        for g in grads:
            if np.max(np.abs(g))>50:
                print(np.max(np.abs(g)))
                sys.exit(1)
        '''
        #sys.exit(1)
        return sess.run(self.W)


def train_once():
    tf.reset_default_graph()
    observation_size=20
    params={}
    params['meta_learning_rate']=0.0001
    params['policy_num_parameters']=observation_size
    meta_policy = meta(params)
    li_W=[]
    li_sigma=[]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(20000):
            state=np.random.random((1,observation_size))
            if t%1000==0:
                print("W: ",sess.run(meta_policy.W))
                print("sigma: ",sess.run(meta_policy.sigma,feed_dict={meta_policy.state:state}))
            action=meta_policy.predict(state,sess)
            #print(action)
            #sys.exit(1)
            G=-np.linalg.norm(state-action)
            meta_policy.update(state,G,action,sess)   
            li_W.append(sess.run(meta_policy.W))
            li_sigma.append(sess.run(meta_policy.sigma,feed_dict={meta_policy.state:state}))
            #print("obsreved return:",G)
            #sys.exit(1)
    return li_W,li_sigma

m=[]
n=[]
for i in range(1):
    li_W,li_sigma=train_once()
    #plt.plot(li)
    m.append(li_W)
    n.append(li_sigma)
plt.plot(np.mean(m,axis=0))
plt.plot(np.mean(n,axis=0))
plt.show()