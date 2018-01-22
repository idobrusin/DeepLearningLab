#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time

import json

import argparse
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import model

from keras import callbacks
# custom modules

from utils import Options, rgb2gray, plot_reward
from simulator import Simulator
from transitionTable import TransitionTable
from datetime import date


# --------------------------------------------------------
# Helper methods
def write_log(callback, names, loss, acc_reward, reward, step):
	logs = loss.copy()
	logs.append(acc_reward)
	logs.append(reward)
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, step)
		callback.writer.flush()


def append_to_hist(state, obs):
	"""
	Add observation to the state.
	"""
	
	for i in range(state.shape[0] - 1):
		state[i, :] = state[i + 1, :]
	state[-1, :] = obs


def print_setup(model_name):
	print("Train agent: ")
	print("  General options: ")
	print("    Display    					: ", opt.disp_on)
	print("    Model  						: ", model_name)
	print("    Log loc.					   	: ", opt.log_path)
	print("    Model Weights loc. 			: ", opt.weights_fil)
	print("    Model Dump loc. 				: ", opt.weights_fil)
	print("    Accumulated Weights plot loc.: ", opt.acc_reward_train_figure_path)
	print("    Model Diagram Location 		: ", opt.model_path)
	print("  Training options: ")
	print("	   Number of steps:   : ", opt.steps)
	print("    Exploration Prob   : ", opt.expl_prob)
	print("    Dual DQN           : ", opt.dual_model)
	print("    Prev. Exp          : ", opt.prev_exp)
	print("    Train interval     : ", opt.train_interval)
	print("    Weight sync interv.: ", opt.weigth_sync_interval)


def train_agent():
	# Sample minibatch
	state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
	
	# Reshape input to fit model input layer
	# state_batch = state_batch.reshape([-1, img_rows, img_cols, opt.hist_len])
	state_batch = state_batch.reshape([-1, img_rows, img_cols, opt.hist_len])
	# next_state_batch = next_state_batch.reshape([-1, img_rows, img_cols, opt.hist_len])
	next_state_batch = next_state_batch.reshape([-1, img_rows, img_cols, opt.hist_len])
	# Predict Q values
	Q = agent.predict_on_batch(state_batch)
	
	if opt.dual_model:
		Q_s_next = target_agent.predict_on_batch(next_state_batch)
	else:
		Q_s_next = agent.predict_on_batch(next_state_batch)
	
	# Sync agent and target agent
	if opt.dual_model and step % opt.weigth_sync_interval == 0:
		print("Copy weight from target to model")
		target_agent.set_weights(agent.get_weights())
	
	# Train agent on batch - iterate over batch for updated Q values
	for i in range(opt.minibatch_size):
		action = np.argmax(action_batch[i])
		
		Q[i, action] = ((1 - terminal_batch[i]) * opt.discount * np.max(Q_s_next[i])) + reward_batch[i]
	# Train agent on new Q
	loss = agent.train_on_batch(state_batch, Q)
	return loss

# --------------------------------------------------------
# CLI
# --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
					help="Name for run / model ")
opt = Options("")  # Temp for params
parser.add_argument("-s", "--steps", help="(Optional) Number of steps to train the model for. Default is 10 ** 6.",
					type=int, default=opt.steps)

args = parser.parse_args()
if args.model:
	model_name = args.model
else:
	model_name = "model" + date.today().strftime("%d_%B_%Y_%I_%M_%p")

# --------------------------------------------------------
# 0. initialization
opt = Options(model_name)
opt.disp_on = False
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

if args.steps:
	opt.steps = args.steps
# --------------------------------------------------------
# Params and Hyperparams
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history,
			   rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

accumulated_reward = 0
accumulated_reward_list = []
# --------------------------------------------------------
# Input layer
image_dimension = opt.cub_siz * opt.pob_siz
img_rows = img_cols = image_dimension
input_shape = [img_rows, img_cols, opt.hist_len]

# --------------------------------------------------------
# Model
agent = model.model(input_shape, opt.vis_model)
target_agent = model.model(input_shape, False)

# --------------------------------------------------------
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len, opt.minibatch_size, maxlen)

# --------------------------------------------------------
# Display
if opt.disp_on:
	win_all = None
	win_pob = None

# --------------------------------------------------------
# Logging and Tensorboard
callback = callbacks.TensorBoard(opt.log_path)
callback.set_model(agent)
data_labels = ['Train loss', 'Train Accuracy', 'Accumulated Reward', "Reward"]

print_setup(model_name)

print("----- Starting training -----")
start_time = time()
print_interval = 0
for step in range(opt.steps):
	if state.terminal or epi_step >= opt.early_stop:
		epi_step = 0
		nepisodes += 1
		
		if step > opt.prev_exp and state.terminal:
			train_agent()
		# reset the game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)
	
	if opt.expl_prob > opt.expl_prob_min:
		opt.expl_prob *= opt.expl_prob_decay  # Decay of random movement probability
	else:
		opt.expl_prob = opt.expl_prob_min
	if step < opt.prev_exp or random.random() <= opt.expl_prob:
		action = random.randrange(opt.act_num)
	else:
		action = np.argmax(agent.predict(state_with_history.reshape(1, img_rows, img_cols, opt.hist_len)))
	
	action_onehot = trans.one_hot_action(action)
	next_state = sim.step(action)
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	# add to the transition table
	trans.add(state_with_history.reshape(-1),
			  action_onehot,
			  next_state_with_history.reshape(-1),
			  next_state.reward,
			  next_state.terminal)
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
	
	# accumulate reward
	accumulated_reward += state.reward
	accumulated_reward_list.append(accumulated_reward)
	
	current_loss = [0.0, 0.0]  # current loss
	if step > opt.prev_exp:
		current_loss = train_agent()

	# Statistics / Progress to stdout
	write_log(callback, data_labels, current_loss, accumulated_reward, state.reward, step)
	if current_loss:
		print_loss = current_loss[0]
		print_accuracy = current_loss[1]
	else:
		print_loss = 0.0
		print_accuracy = 0.0
	print('{:.2f} - Total Step: {:7}- Episode: {:5}, Step: {:3},  Loss: {:.2f}, Accuracy: {:.4f}, Reward: {:.3f}, Acc rew: {:.2f}, Action: {}, Expl:{:.2f}'.format(
		step / opt.steps, step, nepisodes, epi_step, print_loss, print_accuracy, state.reward, accumulated_reward, action, opt.expl_prob))
	
	
	if opt.disp_on:
		if win_all is None:
			plt.subplot(121)
			win_all = plt.imshow(state.screen)
			plt.subplot(122)
			win_pob = plt.imshow(state.pob)
		else:
			win_all.set_data(state.screen)
			win_pob.set_data(state.pob)
		plt.pause(opt.disp_interval)
		plt.draw()
	epi_step = epi_step + 1

elapsed_time = time() - start_time
print("----- Finished training -----")
print("Total time        : {:.2f} seconds".format(elapsed_time))
print("Accumulated reward: ", accumulated_reward_list[-1])
print("Saving weights to : ", opt.weights_fil)

agent.save_weights(opt.weights_fil, overwrite=True)

with open(opt.network_fil, "w") as outfile:
	json.dump(agent.to_json(), outfile)

# Save plot of accumulated reward
plot_reward(accumulated_reward_list, opt.acc_reward_train_figure_path)
