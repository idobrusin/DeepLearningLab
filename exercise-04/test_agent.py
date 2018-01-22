import argparse
from time import time

from agent import DqnAgent
from utils import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_reward(acc_reward, path):
	plt.figure(2)
	plt.plot(acc_reward)
	plt.title('Testing')
	plt.ylabel('Acc. Reward')
	plt.xlabel('Step')
	plt.grid(True)
	plt.savefig(path)
	

def append_to_hist(state, obs):
	"""
	Add observation to the state.
	"""
	for i in range(state.shape[0] - 1):
		state[i, :] = state[i + 1, :]
	state[-1, :] = obs


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
					help="Name for run / model ")
parser.add_argument("-s", "--steps", help="(Optional) Number of steps to train the model for. Default is 1000",
					type=int, default=10000)
parser.add_argument("-ss", "--silent", help="Runs test without displaying the simulation", action="store_false")
args = parser.parse_args()

# --------------------------------------------------------
# 0. initialization
opt = Options()
opt.disp_on = True
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# Display
if opt.disp_on:
	win_all = None
	win_pob = None

opt.disp_on = args.silent
opt.steps = args.steps
run_name = args.model

sess = tf.Session()
agent = DqnAgent(sess, is_duelling_dqn=False)
sess.run(tf.global_variables_initializer())

sess_saver = tf.train.Saver()
sess_saver.restore(sess, opt.data_path_prefix + opt.model_path_prefix + run_name + opt.model_file)

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len, opt.minibatch_size, maxlen)

epi_step = 0
nepisodes = 0
n_completed_episodes = 0
accumulated_reward = 0
accumulated_reward_list = []

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
	
	action = agent.take_q_action(state_with_history)
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
	
	epi_step += 1
	print("Step {}, Action: {}".format(step, action))
	
elapsed_time = time() - start_time
print("----- Finished testing -----")
print("Completed {} games, reached targt {} times".format(nepisodes, n_completed_episodes))
print("{}% of games the target was reached".format(n_completed_episodes / nepisodes))
print("Total time        : {:.2f} seconds".format(elapsed_time))
print("Accumulated reward: ", accumulated_reward_list[-1])

plot_reward(accumulated_reward_list,
			opt.data_path_prefix + opt.model_diagram_path_prefix + run_name + opt.acc_rew_test_file)