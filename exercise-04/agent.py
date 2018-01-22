import random

from tensorflow.python.keras._impl.keras.regularizers import l2

from utils import Options

import tensorflow as tf
import numpy as np

opt = Options()


class DqnAgent:
	def __init__(self, sess, is_duelling_dqn=False):
		# General params and inputs
		self.session = sess  # tf session
		self.state = tf.placeholder(tf.float32, shape=(None, opt.hist_len * opt.state_siz))
		self.action = tf.placeholder(tf.float32, shape=(None, opt.act_num))
		self.action_next = tf.placeholder(tf.float32, shape=(None, opt.act_num))
		self.state_next = tf.placeholder(tf.float32, shape=(None, opt.hist_len * opt.state_siz))
		self.reward = tf.placeholder(tf.float32, shape=(None, 1))
		self.term = tf.placeholder(tf.float32, shape=(None, 1))
		
		# Network and loss
		with tf.variable_scope("Q"):
			self.Q = self.model(self.state)
		with tf.variable_scope("Q", reuse=tf.AUTO_REUSE):
			self.Q_n = self.model(self.state_next)
		
		# Choose next action and init loss tensor
		next_action = self.action_one_hot()
		self.loss = self.Q_loss(self.Q, self.action, self.Q_n, next_action, self.reward, self.term)
		
		# Optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss=self.loss)
		
		# Logging
		tf.summary.scalar("loss", self.loss)
		self.summary = tf.summary.merge_all()
	
	def model(self, state):
		"""
		Neural network for calculating Q-value
		:return: Q-loss tensor
		"""
		with tf.name_scope('Input'):
			input = tf.reshape(state, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len])
		
		with tf.name_scope('Conv1'):
			conv1 = tf.layers.conv2d(
				inputs=input,
				filters=8,
				kernel_size=3,
				padding="same",
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
			)
		with tf.name_scope('Conv2'):
			conv2 = tf.layers.conv2d(
				inputs=conv1,
				filters=8,
				kernel_size=3,
				padding="same",
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
			)
			
		with tf.name_scope('Dense'):
			conv2_flat = tf.reshape(conv2, [-1, int(conv2.shape[1] * conv2.shape[2] * conv2.shape[3])])
			dense1 = tf.layers.dense(
				inputs=conv2_flat,
				units=64,
				kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
				activation=tf.nn.relu,
			)
		
		# Helps stabilizing exploding loss in some cases
		with tf.name_scope('Dropout'):
			dropout = tf.layers.dropout(dense1, rate=0.5, training=True)
		
		# Output with size of number of actions
		with tf.name_scope('Output'):
			output = tf.layers.dense(
				inputs=dropout,
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
				kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
				units=opt.act_num,
				activation=None)
		return output
	
	def train_agent(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
		"""
		Trains agent on minibatch.
		"""
		return self.session.run([self.optimizer, self.summary], feed_dict={
			self.state: state_batch,
			self.state_next: next_state_batch,
			self.action: action_batch,
			self.reward: reward_batch,
			self.term: terminal_batch})
		
	def q_value(self, state):
		"""
		Returns Q-value for given state by performing a forward pass
		:param state: state
		:return: Predicted Q-value
		"""
		return self.session.run(self.Q, feed_dict={self.state: [state]})[0]
	
	def take_action(self, state, exploration_prob, only_random=False):
		"""
		Takes an action from given state.
		:param state: state with history
		:param exploration_prob: probability of random action
		:param only_random: if True, a random action is returned
		:return: action according to q-function or random action
		"""
		if only_random or random.random() < exploration_prob:
			action = random.randrange(opt.act_num)
		else:
			action = self.take_q_action(state)
		return action
	
	def take_q_action(self, state):
		"""
		Returns action according to maximal q-value for a given state
		:param state: state
		:return: index of action
		"""
		q = self.q_value(state.reshape(-1))
		return np.argmax(q)
	
	def action_one_hot(self):
		"""
		Returns one-hot encoded action for given Q-value
		:return: one hot encoded action
		"""
		return tf.one_hot(tf.argmax(self.Q_n, axis=1), 5)
	
	def Q_loss(self, Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
		"""
		All inputs should be tensorflow variables!
		We use the following notation:
			N : minibatch size
			A : number of actions
		Required inputs:
			Q_s: a NxA matrix containing the Q values for each action in the sampled states.
				This should be the output of your neural network.
				We assume that the network implements a function from the state and outputs the
				Q value for each action, each output thus is Q(s,a) for one action
				(this is easier to implement than adding the action as an additional input to your network)
			action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
							(e.g. each row contains only one 1)
			Q_s_next: a NxA matrix containing the Q values for the next states.
			best_action_next: a NxA matrix with the best current action for the next state
			reward: a Nx1 matrix containing the reward for the transition
			terminal: a Nx1 matrix indicating whether the next state was a terminal state
			discount: the discount factor
		"""
		# calculate: reward + discount * Q(s', a*),
		# where a* = arg max_a Q(s', a) is the best action for s' (the next state)
		target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
		# NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
		#       use it as the target for Q_s
		target_q = tf.stop_gradient(target_q)
		# calculate: Q(s, a) where a is simply the action taken to get from s to s'
		selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
		loss = tf.reduce_sum(tf.square(selected_q - target_q))
		return loss