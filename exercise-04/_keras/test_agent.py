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
def write_log(callback, names, logs, acc_reward, step):
	logs.append(acc_reward)
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


def print_setup(model_name, run_name):
	print("Test agent: ")
	print("  Run name: ", run_name)
	print("  General options: ")
	print("    Display     : ", opt.disp_on)
	print("    Model  : ", model_name)
	print("  Training options: ")
	print("	   Number of steps:   : ", opt.steps)


# --------------------------------------------------------
# CLI
# --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path of model to load", required=True)
parser.add_argument("-s", "--steps", help="(Optional) Number of steps to train the model for. Default is 1000",
					type=int, default=1000)
parser.add_argument("-ss", "--silent", help="Runs test without displaying the simulation", action="store_false")

args = parser.parse_args()

model_path = args.model

# --------------------------------------------------------
# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

opt.disp_on = args.silent

if args.steps:
	opt.steps = args.steps

# --------------------------------------------------------
# Input layer
image_dimension = opt.cub_siz * opt.pob_siz
img_rows = img_cols = image_dimension
input_shape = [img_rows, img_cols, opt.hist_len]

# --------------------------------------------------------
# Model
agent = model.model(input_shape)
agent.load_weights(model_path)
if not agent:
	exit(2)

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
# Config
accumulated_reward = 0
accumulated_reward_list = []
n_completed_episodes = 0
nepisodes = 0
epi_step = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

print("----- Starting testing -----")
start_time = time()

for step in range(opt.steps):
	if state.terminal or epi_step >= opt.early_stop:
		if state.terminal:
			n_completed_episodes += 1
		epi_step = 0
		nepisodes += 1
		# reset the game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)
	
	action = np.argmax(agent.predict(state_with_history.reshape(-1, img_rows, img_cols, opt.hist_len)))
	action_onehot = trans.one_hot_action(action)
	next_state = sim.step(action)
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
	
	# accumulate reward
	accumulated_reward += state.reward
	accumulated_reward_list.append(accumulated_reward)
	
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
	
	if step % 100:
		print("Step {}: Accumulated reward= {}".format(step, accumulated_reward))

elapsed_time = time() - start_time
print("----- Finished testing -----")
print("Completed {} games, reached targt {} times".format(opt.steps, n_completed_episodes))
print("{}% of games the target was reached".format(n_completed_episodes / opt.steps))
print("Total time        : {:.2f} seconds".format(elapsed_time))
print("Accumulated reward: ", accumulated_reward_list[-1])

plot_reward(accumulated_reward_list, opt.acc_reward_test_figure_path)
