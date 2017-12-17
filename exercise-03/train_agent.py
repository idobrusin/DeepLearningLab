import json
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam

from utils import Options
from simulator import Simulator
from transitionTable import TransitionTable

import model

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()

sim = Simulator(opt.map_index, opt.cube_size, opt.partial_obs_size, opt.act_num)
trans = TransitionTable(opt.state_size, opt.act_num, opt.history_length,
                        opt.minibatch_size, opt.valid_size,
                        opt.states_file, opt.labels_file)

batch_size = 32
n_epochs = 10
if opt.num_models is None:
	default_weights_file = opt.model_folder + "model0_" + opt.weights_file
	default_network_file = opt.model_folder + "model0_" + opt.network_file

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
# Hint: to ease loading your model later create a model.py file
# where you define your network configuration
######################################

image_dimension = opt.cube_size * opt.partial_obs_size
img_rows = img_cols = image_dimension

train_data = trans.get_train()
valid_data = trans.get_valid()

train_imgs = train_data[0]
train_labels = train_data[1]
valid_imgs = valid_data[0]
valid_labels = valid_data[1]

print("Train: Raw data  shape: ", train_imgs.shape)
print("Train: Raw label shape: ", train_labels.shape)

print("Valid: Raw data  shape: ", valid_imgs.shape)
print("Valid: Raw data  shape: ", valid_labels.shape)

train_imgs = train_imgs.reshape(int(train_imgs.shape[0]), img_rows, img_cols, opt.history_length)
valid_imgs = valid_imgs.reshape(int(valid_imgs.shape[0]), img_rows, img_cols, opt.history_length)
print("Train imgs shape:", train_imgs.shape)
print("Val imgs shape:", valid_imgs.shape)
train_imgs = train_imgs.astype('float32')
valid_imgs = valid_imgs.astype('float32')


def train_agent(weights_file, network_file):
	# Initialize agent
	input_shape = (img_cols, img_rows, opt.history_length)
	layer_params = [
		model.LayerParams(n_filters=32),
		model.LayerParams(n_filters=64),
		model.LayerParams(n_filters=128)
	]
	model_params = model.ModelParams(layer_params=layer_params,
	                                 optimizer=Adam(lr=0.001)
	                                 )
	agent = model.model(input_shape, model_params, visualize_model=True)

	# Train agent
	agent.fit(train_imgs, train_labels, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(valid_imgs, valid_labels))

	print("Metrics:, ", agent.metrics_names)
	score = agent.evaluate(valid_imgs, valid_labels)
	# TODO: find metrics

	print("Test score", score[0])
	print("Test acc", score[1])
	# 2. save your trained model
	agent.save_weights(weights_file, overwrite=True)

	print('Saved weights')
	with open(network_file, "w") as out:
		json.dump(agent.to_json(), out)

def train():
	if opt.num_models is None:
		train_agent(default_weights_file, default_network_file)
	else:
		if type(opt.num_models) is not int:
			print("[ERROR] opt.num_models is not set properly. Must be None or of type int.")
			return
		for i in range(opt.num_models):
			print("Training agent %s of %s bagged agents.".format(i, opt.num_models) )
			wf = opt.model_folder + "model{0}_".format(i) + opt.weights_file
			nf = opt.model_folder + "model{0}_".format(i) + opt.network_file
			train_agent(wf, nf)
	
train()



# https://github.com/yenchenlin/DeepLearningFlappyBird
