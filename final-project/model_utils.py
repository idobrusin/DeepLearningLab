from tensorflow import get_collection, GraphKeys
import tensorflow as tf

default_n_filters = 32
default_kernel_size = 3
default_n_layers = 3
default_units = 64


class ConvLayerParams:

	_activation_funcs = {
		'None': None,
		'relu' : tf.nn.relu,
		'tanh' : tf.tanh
	}

	def __init__(
			self,
			n_filters=default_n_filters,
			kernel_size=default_kernel_size,
			activation='relu',
			padding='valid',
			strides=1,
			kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
			**kwargs
	):
		self.n_filters = n_filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.activation = self._activation_funcs[activation]
		self.padding = padding
		self.kernel_initializer = kernel_initializer
		self.kwargs = kwargs


class DenseLayerParams:
	def __init__(
			self,
			units=default_units,
			activation='relu',
			**kwargs
	):
		self.units = units
		self.activation = activation
		self.kwargs = kwargs


class ModelParams:
	def __init__(
			self,
			layer_params=None,
			optimizer='Adam',
			loss_function='mse',
			metrics=['acc'],
			use_default_conv_layer=False,
			dropout_rate=0.5
	):
		if layer_params is None:
			self.layer_params = []
			if use_default_conv_layer:
				new_layer = ConvLayerParams
			else:
				new_layer = DenseLayerParams
			for i in range(default_n_layers):
				self.layer_params.append(new_layer())
		else:
			self.layer_params = layer_params
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.metrics = metrics
		self.dropout_rate = dropout_rate


def get_variables_by_name(name):
	return get_collection(GraphKeys.TRAINABLE_VARIABLES, name)


def copy_to_target_network(source_network_name, target_network_name):
	target_network_update = []
	for v_source, v_target in zip(get_variables_by_name(source_network_name),
								  get_variables_by_name(target_network_name)):
		# this is equivalent to target = source
		update_op = v_target.assign(v_source)
		target_network_update.append(update_op)
	print("Finished copying")
	return tf.group(*target_network_update)
