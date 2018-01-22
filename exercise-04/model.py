from keras import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.regularizers import l2
from keras.utils import plot_model
import tensorflow as tf

default_n_filters = 32
default_kernel_size = 3
default_n_layers = 3


class LayerParams:
	def __init__(
			self,
			n_filters=default_n_filters,
			kernel_size=default_kernel_size,
			activation='relu',
			padding='same',
	):
		self.n_filters = n_filters
		self.kernel_size = kernel_size
		self.activation = activation
		self.padding = padding


class ModelParams:
	def __init__(
			self,
			layer_params=None,
			optimizer='Adam',
			loss_function='categorical_crossentropy',
			metrics=['acc']
	):
		if layer_params is None:
			self.layer_params = []
			for i in range(default_n_layers):
				self.layer_params.append(LayerParams())
		else:
			self.layer_params = layer_params
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.metrics = metrics
		
		
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


		
def model(input_shape, model_params=None, visualize_model=False):
	"""
	Model for training agent for grid environment
	:param: rows - number of rows in a single image
	:param: cols - number of cols in a single image
	:param: history_length - number of frames previous to the current one
	:return: raw keras model of agent
	"""
	
	# LINKS:
	# Nice visualizations of deep q networks
	#https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
		
	# Example code with links to DQN Papers
	# https://github.com/carpedm20/deep-rl-tensorflow
	
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# NOTE:
	# You should prepare your network training here. I suggest to put this into a
	# class by itself but in general what you want to do is roughly the following
	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	"""
	# setup placeholders for states (x) actions (u) and rewards and terminal values
	x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
	u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
	ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
	xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
	r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
	term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

	# get the output from your network
	Q = my_network_forward_pass(x)
	Qn =  my_network_forward_pass(xn)

	# calculate the loss
	loss = Q_loss(Q, u, Qn, ustar, r, term)

	# setup an optimizer in tensorflow to minimize the loss
	"""
	
	# input_shape = (rows, cols, history_length)
	# assert (model_params)
	
	model = Sequential()
	layer_params = model_params.layer_params
	
	model.add(Convolution2D(
		filters=layer_params[0].n_filters,
		kernel_size=layer_params[0].kernel_size,
		activation=layer_params[0].activation,
		padding=layer_params[0].padding,
		input_shape=input_shape,
		name='ConvLayer0'
	))
	model.add(MaxPooling2D(name="MaxPool0"))
	
	for i, param in enumerate(layer_params):
		if i == 0:
			continue
		
		model.add(Convolution2D(
			filters=param.n_filters,
			kernel_size=param.kernel_size,
			activation=param.activation,
			padding=param.padding,
			name='ConvLayer' + str(i)
		))
		
		model.add(MaxPooling2D(name="MaxPool" + str(i)))
	
	model.add(Flatten())
	model.add(Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
					kernel_initializer='random_uniform', bias_initializer='zeros'))
	model.add(Dense(5, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='softmax',
					kernel_initializer='random_uniform', bias_initializer='zeros'))
	
	model.compile(
		optimizer=model_params.optimizer,
		loss=model_params.loss_function,
		metrics=model_params.metrics
	)
	
	if visualize_model:
		print("Printing model diagram")
		plot_model(model, to_file='./data/images/model.png', show_shapes=True)
	
	return model
