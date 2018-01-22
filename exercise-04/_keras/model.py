from keras import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from utils import Options

default_n_filters = 32
default_kernel_size = 3
default_n_layers = 3


def load_existing(model_path):
	try:
		agent = load_model(model_path)
	except Exception as e:
		print("[WARN] Could not load model with name {0}, encountered exception {1}".format(model_path, e))
		return False
	return agent


def model(input_shape, model_name='', visualize_model=False):
	opt = Options()
	
	model = Sequential()
	model.add(Convolution2D(
		input_shape=input_shape,
		filters=default_n_filters,
		kernel_size=default_kernel_size,
		activation='relu',
		padding='same',
		name='ConvLayer1'
	))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(opt.act_num))
	model.compile(loss='mse', metrics=['acc'], optimizer=Adam(lr=1e-6))
	
	if visualize_model:
		print("Saving model diagram")
		plot_model(model, to_file=opt.model_path.format(model_name), show_shapes=True)
	return model

