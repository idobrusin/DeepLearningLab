import argparse

from agent import DqnAgent
from simulator import Simulator
from transitionTable import TransitionTable
from utils import Options, rgb2gray

from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def plot_reward(acc_reward, path):
	plt.figure(2)
	plt.plot(acc_reward)
	plt.title('Training')
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


# --------------------------------------------------------
# CLI
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str,
					help="Name for run / model ")
args = parser.parse_args()
run_name = args.model

# --------------------------------------------------------
# 0. initialization
opt = Options()
opt.disp_on = False
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

sess = tf.Session()
agent = DqnAgent(sess, is_duelling_dqn=False)
sess.run(tf.global_variables_initializer())

sess_saver = tf.train.Saver()
writer = tf.summary.FileWriter("%s/%s" % (opt.data_path_prefix + opt.log_path_prefix, run_name), sess.graph)

# --------------------------------------------------------
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
						opt.minibatch_size, maxlen)

if opt.disp_on:
	win_all = None
	win_pob = None

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
# steps = 1 * 10 ** 6
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

accumulated_reward = 0
accumulated_reward_list = []
loss = 0

print("----- Starting training -----")
start_time = time()
finished_ep_count = 0
for step in range(opt.steps):
	if state.terminal or epi_step >= opt.early_stop:
		if epi_step < opt.early_stop:
			finished_ep_count += 1
		
		print(
			'{:.2f} % - Total Step: {:7}- Episode: {:5}, Step: {:3}, Finished:{}, Loss:{}'.format(
				step / opt.steps, step, nepisodes, epi_step, finished_ep_count, loss))
		epi_step = 0
		nepisodes += 1
		# reset the game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)
	epi_step += 1
	
	# Let agent move according to q-function or randomly
	if step < opt.prior_experience:
		action = agent.take_action(state_with_history, opt.exploration_prob, True)
	else:
		# Adjust random exploration factor
		if opt.exploration_prob > opt.exploration_prob_min:
			opt.exploration_prob *= opt.exploration_decay  # Decay of random movement probability
		else:
			opt.exploration_prob = opt.exploration_prob_min
		action = agent.take_action(state_with_history, opt.exploration_prob)
	
	action_onehot = trans.one_hot_action(action)
	next_state = sim.step(action)
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	# add to the transition table
	trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward,
			  next_state.terminal)
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
	
	accumulated_reward += state.reward
	accumulated_reward_list.append(accumulated_reward)
	
	# Sample minibatch
	state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
	
	# Train agent after the agent moved around randomly.
	if step > opt.prior_experience:
		# Train agent
		_, summary= agent.train_agent(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch)
		writer.add_summary(summary, step)

		if step % 1000 == 0:
			# Save weights
			save_path = sess_saver.save(sess, opt.data_path_prefix + opt.model_path_prefix + run_name + opt.model_file)
			print("Model saved in file: %s" % save_path)
	
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

elapsed_time = time() - start_time
print("----- Finished training -----")
print("Total time        : {:.2f} seconds".format(elapsed_time))
print("Accumulated reward: ", accumulated_reward_list[-1])

# Save plot of accumulated reward
plot_reward(accumulated_reward_list,
			opt.data_path_prefix + opt.model_diagram_path_prefix + run_name + opt.acc_rew_train_file)