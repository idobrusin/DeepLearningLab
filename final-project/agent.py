import random

from model_utils import ConvLayerParams, DenseLayerParams, ModelParams
import tensorflow as tf
import numpy as np

from utils import Options, DqnAlgorithm


class DqnAgent:
	def __init__(self, sess, opt, model_params=None, restore_model=False):
		'''
		sess is the tensorflow session within which the instance of DqnAgent will live.
		opt is an instance of utils.Options
		model_params is an instance of model_utils.ModelParams
		is_duelling_dqn specifies whether or not the target networks functionality is enabled
		restore_model specifies that the network will be restored from a previous saved session
		'''
		# General params and inputs
		self.session = sess  # tf session
		self.opt = opt
		self.state = tf.placeholder(tf.float32, shape=(None, opt.hist_len * opt.state_siz))
		self.action = tf.placeholder(tf.float32, shape=(None, opt.act_num))
		self.action_next = tf.placeholder(tf.float32, shape=(None, opt.act_num))
		self.state_next = tf.placeholder(tf.float32, shape=(None, opt.hist_len * opt.state_siz))
		self.reward = tf.placeholder(tf.float32, shape=(None, 1))
		self.term = tf.placeholder(tf.float32, shape=(None, 1))
		self.exploration_prob = opt.exploration_prob
		self.exploration_decay = opt.exploration_decay

		# if restore_model:
		# 	self.mode = opt.TEST
		# else:
		# 	self.mode = opt.TRAIN
		
		self.layer_params, self.metrics, self.loss_function, self.dropout_rate = self.set_model_params(model_params)
		self.Q, self.Q_n = self.init_model()
		
		print("Creating agent with {} algorithm".format(opt.chosen_algo))
		
		if self.opt.chosen_algo == DqnAlgorithm.DDQN or self.opt.chosen_algo == DqnAlgorithm.DQN_DUELLING:
			# DDQN uses the source network to choose an action
			print("Source network choose action")
			next_action = self.action_one_hot(self.Q)
		else:
			print("Target network chooses action")
			next_action = self.action_one_hot(self.Q_n)
		
		with tf.variable_scope("Loss"):
			self.loss = self.Q_loss(self.Q, self.action, self.Q_n, next_action, self.reward, self.term)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate).minimize(loss=self.loss)
		
		# Logging
		tf.summary.scalar(opt.SUMMARY_LOSS_NAME, self.loss)
		self.summary = tf.summary.merge_all()

	def init_model(self):
		"""
		Initialises network depending on algorithm to be used for the network architecture
		"""
		
		if self.opt.chosen_algo == DqnAlgorithm.DQN:
			return self._init_DQN()
		elif self.opt.chosen_algo == DqnAlgorithm.DQN_TARGET or self.opt.chosen_algo == DqnAlgorithm.DDQN:
			return self._init_TGT_NTWRKS()
		elif self.opt.chosen_algo == DqnAlgorithm.DQN_DUELLING:
			return self._init_DUELING()
		
	def _init_DQN(self):
		"""
		Normal DQN. Uses only one NN to generate Q-values, choose the next best action, and generate loss.
		"""
		with tf.variable_scope("DQN"):
			Q = self.model(self.state)
		with tf.variable_scope("DQN", reuse=tf.AUTO_REUSE):
			Q_n = self.model(self.state_next)
		return Q, Q_n
	
	def _init_TGT_NTWRKS(self):
		"""
		DQN with Target Networks. Contains a Source network which is actively updated every training step
		and a Target network which is periodically updated by copying the weights from the source
		network. The Bellman Equation uses the the target network to both choose the best action in the
		next state and to evaluate it's Q-value.
		
		Double DQN. Similar to DQN with Target Networks in the sense that it employs a similar Source and
		Target network setup, but the difference is that in the Bellman Equation, the best action in the
		subsequent state is chosen using the Source Network whereas the Q-value of this action in that
		state is determined using the Target Network.
		"""
		with tf.variable_scope("SourceNetwork"):
			Q = self.model(self.state)
		with tf.variable_scope("TargetNetwork"):
			Q_n = self.model(self.state_next)
		return Q, Q_n

	def _init_DUELING(self):
		"""
		Dueling DQN. Builds up on the
		"""
		with tf.variable_scope("SourceNetwork"):
			Q = self.model(self.state, use_duelling=True)
		with tf.variable_scope("TargetNetwork"):
			Q_n = self.model(self.state_next, use_duelling=True)
		return Q, Q_n

	def set_model_params(self, model_params):
		# if self.mode == self.opt.TEST:
		# 	return
		if model_params is None:
			print("[WARN] No model params specified, using default values of network architecture")
			model_params = ModelParams()
			
		return model_params.layer_params, model_params.metrics, model_params.loss_function, model_params.dropout_rate

	def model(self, state, use_m=None, use_duelling=False):
		# if self.mode == self.opt.TEST:
		# 	return tf.get_default_graph().get_operation_by_name("NonDuelling/" +self.opt.OUTPUT_TENSOR_NAME + "/BiasAdd")
		
		if use_m is not None:
			mtype = use_m
		else:
			mtype  = self.opt.mtype
		if mtype == 'Conv':
			return self._convModel(state, use_duelling)
		elif mtype == 'Dense':
			return self._denseModel(state)
		elif mtype == 'ConvPool':
			return self._convModel_withPooling(state)
		else:
			raise Exception('Unknown model type {0}'.format(mtype))
	
	def _convModel(self, state, use_duelling=False):
		opt = self.opt

		layer_count = 0
		layers = []
	
		layers.append(tf.reshape(state, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len], name='Input_Layer'))
		layer_count += 1
		# Add convolutional layers
		for layer_param in self.layer_params:
			try:
				layers.append(
					tf.layers.conv2d(
						inputs=layers[-1],
						filters=layer_param.n_filters,
						kernel_size=layer_param.kernel_size,
						padding=layer_param.padding,
						activation=layer_param.activation,
						kernel_initializer=layer_param.kernel_initializer,
						name="Conv{0}".format(layer_count),
						strides=layer_param.strides,
					)
				)
				layer_count += 1
			except Exception as e:
				print(e)
				exit(-1)
				
		if use_duelling:
			last_layer = tf.layers.conv2d(inputs=layers[-1], filters=32, kernel_size=3, padding='valid')
			
			# Split along number of filters (TBD ???) - [20,20, 512] -> [20, 20, 256]
			self.advantage_conv, self.value_conv = tf.split(last_layer, 2, 3)
			# self.advantage_conv, self.value_conv = tf.split(3, 2, last_layer)
			
			with tf.variable_scope("Advantage"):
				flat = tf.layers.flatten(self.advantage_conv)
				adv_dense = tf.layers.dense(
						inputs=flat,
						kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
						units=opt.act_num,
						activation=tf.nn.relu,
						kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
						name="Dense"
				)
			
			with tf.variable_scope("Value"):
				value_flat = tf.layers.flatten(self.value_conv)
				value_dense = tf.layers.dense(
						inputs=value_flat,
						kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
						units=1,
						activation=tf.nn.relu,
						kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
						name="Dense"
				)

			# Combine them together to get our final Q-values.
			output = value_dense + tf.subtract(adv_dense, tf.reduce_mean(adv_dense, axis=1, keep_dims=True))
			return output
		else:
			# Add a flattening layer
			last_layer = layers[-1]
			flat = tf.reshape(last_layer, [-1, int(last_layer.shape[1] * last_layer.shape[2] * last_layer.shape[3])])
			layers.append(
				tf.layers.dense(
					inputs=flat,
					units=128,
					activation=tf.nn.relu,
					# activation=tf.sigmoid
					kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
				)
			)
			layer_count += 1
			# Add a dropout layer
			# Helps stabilizing exploding loss in some cases
			layers.append(tf.layers.dropout(layers[-1], rate=self.dropout_rate, training=True))
			layer_count += 1
			# Add an output layer (NOTE: No activation function!)
			# Output with size of number of actions
			layers.append(
				tf.layers.dense(
					inputs=layers[-1],
					kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
					units=opt.act_num,
					activation=None,
					name=opt.OUTPUT_TENSOR_NAME
				)
			)
			return layers[-1]

	def _denseModel(self, state):
		opt = self.opt
		layer_count = 0
		layers = []

		layers.append(tf.reshape(state, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len], name='Input_Layer'))
		layer_count += 1
		# Add all dense layers
		for layer_param in self.layer_params:
			if type(layer_param) is not DenseLayerParam:
				raise Exception("Received unexpected layer parameter type at layer_params index {0}, expected ConvLayerParams, received {1}.".format(layer_count, layer_param.__class__.__name__))
			layers.append(
				tf.layers.dense(
					inputs=layers[layer_count-1],
					units=layer_param.units,
					activation=layer_param.activation,
					name="Dense{0}".format(layer_count),
#					**layer_param.kwargs
				)
			)
			layer_count += 1
		# Add a dropout layer
		# Helps stabilizing exploding loss in some cases
		layers.append(tf.layers.dropout(layers[layer_count-1], rate=self.dropout_rate, training=True))
		layer_count += 1
		# Add an output layer
		# Output with size of number of actions
		layers.append(
			tf.layers.dense(
				inputs=layers[layer_count-1],
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
				units=opt.act_num,
				activation=None,
				name=opt.OUTPUT_TENSOR_NAME
			)
		)
		return layers[-1]

	def _convModel_withPooling(self, state):
		opt = self.opt

		layer_count = 0
		layers = []
	
		layers.append(tf.reshape(state, [-1, opt.pob_siz * opt.cub_siz, opt.pob_siz * opt.cub_siz, opt.hist_len], name='Input_Layer'))
		layer_count += 1
		# Add convolutional layers
		for layer_param in self.layer_params:
	#		if type(layer_param) is not ConvLayerParams:
	#			raise Exception("Received unexpected layer parameter type at layer_params index {0}, expected ConvLayerParams, received {1}.".format(layer_count, layer_param.__class__.__name__))
			try:
				layers.append(
					tf.layers.conv2d(
						inputs=layers[-1],
						filters=layer_param.n_filters,
						kernel_size=layer_param.kernel_size,
						padding=layer_param.padding,
						activation=layer_param.activation,
						kernel_initializer=layer_param.kernel_initializer,
						name="Conv{0}".format(layer_count),
	#					**layer_param.kwargs
					)
				)
#	Pooling layer. Arguments: 
#		input - input tensor, here, 4D [batch, height, width, channels]
#		window_shape - pooling window size per pooling dimension
#		strides - pooling window stride in each pooling dimension
#		padding - 'VALID' or 'SAME'
		
			#	ENABLE THIS TO VISUALIZE PARTIALLY CONSTRUCTED GRAPH
			#	opt.summary_writer.add_graph(tf.get_default_graph())
			#	opt.summary_writer.flush()

				layers.append(
					tf.nn.pool(
						input=layers[-1],
						window_shape=[2, 2],
						pooling_type='AVG',
						strides=[2, 2],
						padding='VALID'
					)
				)
				layer_count += 2
			except Exception as e:
				print(e)
				exit(-1)
		# Add a flattening layer
		last_layer = layers[-1]
		flat = tf.reshape(last_layer, [-1, int(last_layer.shape[1] * last_layer.shape[2] * last_layer.shape[3])])
		layers.append(
			tf.layers.dense(
				inputs=flat,
				units=64,
				activation=tf.nn.relu,
			)	
		)
		layer_count += 1
		# Add a dropout layer
		# Helps stabilizing exploding loss in some cases
		layers.append(tf.layers.dropout(layers[layer_count-1], rate=self.dropout_rate, training=True))
		layer_count += 1
		# Add an output layer (NOTE: No activation function!)
		# Output with size of number of actions
		layers.append(
			tf.layers.dense(
				inputs=layers[layer_count-1],
				kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
				units=opt.act_num,
				activation=None,
				name=opt.OUTPUT_TENSOR_NAME
			)
		)
		return layers[-1]

	def train_agent(self, state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
		"""
		Trains agent on minibatch.
		"""
		return self.session.run([self.optimizer, self.summary, self.loss], feed_dict={
			self.state: state_batch,
			self.state_next: next_state_batch,
			self.action: action_batch,
			self.reward: reward_batch,
			self.term: terminal_batch})
		
	def q_value(self, state, Q):
		"""
		Returns Q-value for given state by performing a forward pass
		:param state: state
		:return: Predicted Q-value
		"""
		return self.session.run(Q, feed_dict={self.state: [state]})[0]
	
	def choose_action(
		self,
		state, 
		use_random=True, 
		only_random=False, 
		use_Qn=False
	):
		"""
		Takes an action from given state.
		:param state: state with history
		:param use_random: Enable random exploration
		:param exploration_prob: probability of random action
		:param only_random: if True, a random action is returned
		:param use_Qn: If True, the Q_n tensor is used to choose an action
		:return: action according to q-function or random action
		"""
		# Adjust the random exploration factor
		if use_random:
			if self.exploration_prob >self.opt.exploration_prob_min:
				self.exploration_prob -= self.exploration_decay
			else:
				self.exploration_prob = self.opt.exploration_prob_min

		# Decide whether to use random exploration or follow the policy
		if only_random or use_random and random.random() < self.exploration_prob:
			action = random.randrange(self.opt.act_num)
		else:
			action = self.choose_q_action(state)
		return action
	
	def choose_q_action(self, state):
		"""
		Returns action according to maximal q-value for a given state
		:param state: state
		:param use_Qn: If True, the Q_n tensor is used for all computations
		:return: index of action
		"""
		q = self.q_value(state.reshape(-1), self.Q)
		return np.argmax(q)
	
	def action_one_hot(self, Q):
		"""
		Returns one-hot encoded action for given Q-value
		:return: one hot encoded action
		"""
		return tf.one_hot(tf.argmax(Q, axis=1), self.opt.act_num, name=self.opt.ONE_HOT_ACTION_TENSOR_NAME) \
	#if self.mode == self.opt.TRAIN else tf.get_default_graph().get_operation_by_name(self.opt.ONE_HOT_ACTION_TENSOR_NAME)
	
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
	#	if self.mode == self.opt.TEST:
	#		return tf.get_default_graph().get_operation_by_name(self.opt.LOSS_TENSOR_NAME)
		# calculate: reward + discount * Q(s', a*),
		# where a* = arg max_a Q(s', a) is the best action for s' (the next state)
		target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
		# NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
		#       use it as the target for Q_s
		target_q = tf.stop_gradient(target_q)
		# calculate: Q(s, a) where a is simply the action taken to get from s to s'
		selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
		loss = tf.reduce_sum(tf.square(selected_q - target_q), name=self.opt.LOSS_TENSOR_NAME)
		return loss
