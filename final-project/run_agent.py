import argparse

from agent import DqnAgent
from simulator import Simulator
from transitionTable import TransitionTable
from utils import Options, rgb2gray, DqnAlgorithm
# from model_utils import ConvLayerParams, ModelParams
from model_utils import *
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


# We will load all our hard-coded internal defaults from here
opt = Options()

# --------------------------------------------------------
# CLI
from_file = False

def handle_cli(args):
	opt.prefix = args.prefix
	opt.set_algo(args.algo)
	opt.steps = args.steps
	opt.mtype = args.mtype
	opt.hist_len = args.hlength
	print('[INFO] Parsed parameters from the Command Line.')

def handle_cfg(args):
	opt.load_config_file(args.file)
	opt.rtype = args.rtype
	from_file = True

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(
	title='Mode of Operation',
	help='Choose how the model and simulation parameters are provided.'
)

file_sub = subparsers.add_parser('cfg', help='Subcommand to use a configuration file.')

file_sub.add_argument(
	"file", 
	type=str, 
	help="Name of configuration file to use for generating the model and simulation. \
Config files must be placed in the configuration directory. Priority of settings: \
1-(Highest) Config File...\
2-Command Line settings...\
3-(Lowest) Default settings" 
)

file_sub.add_argument(
	"rtype", 
	help="Type of run, available values: TRAIN - train an agent, TEST - test an agent.", 
	type=str, 
	choices=opt.supported_rtypes
)


file_sub.set_defaults(func=handle_cfg)

cli_sub = subparsers.add_parser('cli', help='Subcommand to read configuration from the CLI.')

cli_sub.add_argument(
	"prefix",
	type=str,
	help="Prefix name for run / model "
)

cli_sub.add_argument(
	"-s", 
	"--steps", 
	help="Number of steps.", 
	type=int, 
	default=10000
)

cli_sub.add_argument(
	"-l", 
	"--hlength", 
	help="History length.", 
	type=int, 
	default=4
)

cli_sub.add_argument(
	"-m", 
	"--mtype", 
	help="Type of model to use. Currently supported options:Conv - Convolution Layers without Pooling, ConvPool - Convolution Layers with Pooling, Dense - Dense layers.", 
	type=str, 
	default='Conv', 
	choices=opt.supported_mtypes
)

cli_sub.add_argument(
	"rtype", 
	help="Type of run, available values: TRAIN - train an agent, TEST - test an agent.", 
	type=str, 
	choices=opt.supported_rtypes
)

cli_sub.add_argument(
	"algo", 
	help="Algorithm. Choices:\n\t1- Normal DQN.\n\t2- DQN With Target Networks.\n\t3- Double DQN.\n\t4- Duelling DDQN.", 
	type=int,
	choices=range(1, 5)
)

cli_sub.set_defaults(func=handle_cli)

# parser.add_argument(
# 	"rtype", 
# 	help="Type of run, available values: TRAIN - train an agent, TEST - test an agent.", 
# 	type=str, 
# 	choices=opt.supported_rtypes
# )

parser.add_argument(
	"-V", 
	"--visualize", 
	help="Visualize simulation.", 
	action="store_true"
)

parser.add_argument(
	"--plot_output", 
	help="Enable plotting of the final results", 
	action="store_true"
)

parser.add_argument(
	"-G",
	"--graph",
	help="Debugging option. Terminates run after generating graph.",
	action="store_true"
)

args = parser.parse_args()
args.func(args)
'''if not args.cfg_file:
	opt.prefix = args.prefix
	opt.set_algo(args.algo)
	opt.steps = args.steps
	opt.mtype = args.mtype
	opt.rtype = args.rtype
	opt.hist_len = args.hlength
	from_file = False
else:
	opt.load_config_file(args.file)
	from_file = True'''

opt.disp_on = args.visualize
plot = args.plot_output
model_name = opt.generate_model_name()
run_name = opt.generate_run_name()

# Environment setup
opt.disp_on = args.visualize
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

if opt.rtype == opt.supported_rtypes[0]:
	opt.training = True
else:
	opt.testing = True

# Default model architecture
layer_params = [
	ConvLayerParams(
		n_filters=16,
		strides=2,
		kernel_size=3
	),
	ConvLayerParams(
		n_filters=32,
		strides=2,
		kernel_size=3
	)
]
model_params = ModelParams(layer_params=layer_params)

# Create agent object
sess = tf.Session()

# Add a writer, which changes depending on the type of run, i.e. training or testing
if opt.training:
	opt.summary_writer = writer = tf.summary.FileWriter("%s/%s" % (opt.data_path_prefix + opt.log_path_prefix, run_name))
else:
	opt.summary_writer = writer = tf.summary.FileWriter("%s/%s_%s" % (opt.data_path_prefix + opt.log_path_prefix, run_name, "TEST"))

if from_file:
	agent = DqnAgent(sess, opt, opt.get_model_params())
else:
	agent = DqnAgent(sess, opt, model_params)

# print('[DEBUG] Graph created.')

writer.add_graph(sess.graph)

if opt.training:
	sess_saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
elif opt.testing:
	model_file_path = opt.data_path_prefix + opt.model_path_prefix + run_name + opt.model_file
	# 	sess_saver = tf.train.import_meta_graph(model_file_path + ".meta")
	sess_saver = tf.train.Saver()
	sess_saver.restore(sess, model_file_path)
else:
	print("Dude, wtf?")
	exit(-1)

# Generates a graph to be visualized in TensorBoard. Helps in debugging.
writer.flush()
if args.graph:
	exit(0)

# Training mode specific initialization
if opt.training:
	# setup a large transitiontable that is filled during training
	maxlen = opt.maxlen
	trans = TransitionTable(
		opt.state_siz,
		opt.act_num,
		opt.hist_len,
		opt.minibatch_size,
		maxlen
	)

if opt.disp_on:
	win_all = None
	win_pob = None

# Initialize simulation
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

accumulated_reward = 0
accumulated_reward_list = []
loss = 0

start_time = time()
finished_ep_count = 0
avg_steps_per_finished_ep = 0
steps_per_finished_ep = []

# Additional metrics as tf summaries
steps_per_finished_ep_ph = tf.placeholder(tf.int32, [])
steps_per_finished_ep_summary = tf.summary.scalar(opt.SUMMARY_STEP_PER_EPISODE, steps_per_finished_ep_ph)

finished_ep_count_ph = tf.placeholder(tf.int32, [])
finished_ep_count_summary = tf.summary.scalar(opt.SUMMARY_NUM_FINISHED_EP, finished_ep_count_ph)

accumulated_reward_ph = tf.placeholder(tf.float32, [])
accumulated_reward_summary = tf.summary.scalar(opt.SUMMARY_ACC_REWARD, accumulated_reward_ph)

avg_steps_per_finished_ep_ph = tf.placeholder(tf.float32, [])
avg_steps_per_finished_ep_summary = tf.summary.scalar(opt.SUMMARY_STEP_PER_EPISODE_AVG, avg_steps_per_finished_ep_ph)

# Start simulation
for step in range(opt.steps):
	if state.terminal or epi_step >= opt.early_stop:
		if state.terminal:
			finished_ep_count += 1
			steps_per_finished_ep.append(epi_step)
			avg_steps_per_finished_ep = avg_steps_per_finished_ep + (
				(epi_step - avg_steps_per_finished_ep) / finished_ep_count)
			
			summ = sess.run([
					steps_per_finished_ep_summary,
					finished_ep_count_summary,
					accumulated_reward_summary,
					avg_steps_per_finished_ep_summary],
				feed_dict={
					steps_per_finished_ep_ph: epi_step,
					finished_ep_count_ph: finished_ep_count,
					accumulated_reward_ph: accumulated_reward,
					avg_steps_per_finished_ep_ph: avg_steps_per_finished_ep
				}
			)
			writer.add_summary(summ[0], finished_ep_count)  # don't use step for better plots
			writer.add_summary(summ[1], step)
			writer.add_summary(summ[2], step)
			writer.add_summary(summ[3], finished_ep_count)  # don't use step for better plots
		
		print(
			'{:.2f} % - Total Step: {:7}- Episode: {:5}, Step: {:3}, Finished:{}, Loss:{}'.format(
				100.0 * step / opt.steps, step, nepisodes, epi_step, finished_ep_count, loss))
		
		epi_step = 0
		nepisodes += 1
		
		# reset the game
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)
	
	epi_step += 1
	
	if opt.training:
		# Let agent move according to q-function or randomly
		if step < opt.prior_experience:
			action = agent.choose_action(state_with_history, only_random=True)
		else:
			action = agent.choose_action(
				state_with_history,
				only_random=False,
				use_Qn=False
			)
	else:
		# If we aren't training, then we're testing.
		# REMINDER: choose_action() includes random exploration,
		# whereas choose_q_action directly uses the relevant Q-value to determine an action
		action = agent.choose_q_action(state_with_history)
	
	next_state = sim.step(action)
	# append to history
	append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
	if opt.training:
		action_onehot = trans.one_hot_action(action)
		# add to the transition table
		trans.add(
			state_with_history.reshape(-1),
			action_onehot,
			next_state_with_history.reshape(-1),
			next_state.reward,
			next_state.terminal
		)
	# mark next state as current state
	state_with_history = np.copy(next_state_with_history)
	state = next_state
	
	accumulated_reward += state.reward
	accumulated_reward_list.append(accumulated_reward)
	
	if opt.training:
		# Sample minibatch
		state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
		
		# Train agent after the agent moved around randomly.
		if step > opt.prior_experience:
			# Train agent
			_, summary, loss = agent.train_agent(state_batch, action_batch, next_state_batch, reward_batch,
												 terminal_batch)
			writer.add_summary(summary, step)
		
		if step % 1000 == 0:
			# Save weights
			save_path = sess_saver.save(sess, opt.data_path_prefix + opt.model_path_prefix + run_name + opt.model_file)
			print("Model saved in file: %s" % save_path)
			
			if agent.opt.chosen_algo != DqnAlgorithm.DQN:
				sess.run(copy_to_target_network("SourceNetwork", "TargetNetwork"))
	
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
	if opt.testing:
		print("Step {}, Action: {}".format(step, action))

elapsed_time = time() - start_time
print("----- Finished run-----")
print("Total time        : {:.2f} seconds".format(elapsed_time))
print("Accumulated reward: ", accumulated_reward_list[-1])
if opt.testing:
	print("Completed {} games, reached target {} times".format(nepisodes, finished_ep_count))
	print("{}% of games the target was reached".format(100.0 * finished_ep_count / nepisodes))
	print("Average steps per finished episode", np.average(steps_per_finished_ep))
# Save plot of accumulated reward
# plot_reward(accumulated_reward_list,
#			opt.data_path_prefix + opt.model_diagram_path_prefix + run_name + opt.acc_rew_train_file)

writer.close()
