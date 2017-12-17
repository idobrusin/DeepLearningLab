import numpy as np
import matplotlib
from keras.optimizers import Adam

from functools import partial
from collections import Counter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# custom modules
from utils import Options, rgb2gray
from simulator import Simulator
import model


# 0. initialization (initialize before util methods to have access to options)
opt = Options()
sim = Simulator(opt.map_index, opt.cube_size, opt.partial_obs_size, opt.act_num)
# opt.disp_on = False


def append_history(state_hist, new_state):
	"""
	Appends given new_state to state history, moving previous states in the array
	:param state_hist: history of states
	:param new_state: new state to append to history
	"""
	for i in range(state_hist.shape[0] - 1):
		state_hist[i, :] = state_hist[i + 1, :]
	state_hist[-1, :] = new_state


def image_to_state(img, state_size=opt.state_size):
	return rgb2gray(img).reshape(state_size)


def reset_game():
	# start a new game
	state = sim.newGame(opt.target_y, opt.target_x)
	# Initialize history
	state_hist = np.zeros((opt.history_length, opt.state_size))
	append_history(state_hist, image_to_state(state.pob))

	return state, state_hist


# Initialize agent
image_dimension = opt.cube_size * opt.partial_obs_size
img_rows = img_cols = image_dimension

input_shape = (img_cols, img_rows, opt.history_length)

layer_params = [
	model.LayerParams(n_filters=32),
	model.LayerParams(n_filters=64),
	model.LayerParams(n_filters=128)
]
model_params = model.ModelParams(layer_params=layer_params, optimizer=Adam(lr=0.001))

def test_agent():
	agents = []
	if opt.num_models is None:
		wf = opt.model_folder + "model4_" + opt.weights_file
		agent = model.model(input_shape, model_params)
		agent.load_weights(wf)
		agents.append(agent)
		print("Loaded single agents")

	else:
		for i in range(opt.num_models):
			wf = opt.model_folder + "model{0}_".format(i) + opt.weights_file
			agent = model.model(input_shape, model_params)
			agent.load_weights(wf)
			agents.append(agent)
		print("Loaded multiple agents")

	# 1. control loop
	if opt.disp_on:
		win_all = None
		win_pob = None
	epi_step = 0  # #steps in current episode
	nepisodes = 0  # total #episodes executed
	nepisodes_solved = 0
	action = 0  # action to take given by the network

	# start a new game
	state, state_hist = reset_game()

	for step in range(opt.eval_steps):

		# check if episode ended
		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			if state.terminal:
				nepisodes_solved += 1
			# start a new game
			reset_game()

		else:
			epi_step += 1
			# take action
			action = predict(agents, state_hist.reshape(1, img_rows, img_cols, opt.history_length))
			# action = np.argmax(agent.predict(state_hist.reshape(1, img_rows, img_cols, opt.history_length)))
			state = sim.step(action)

			# append to history
			append_history(state_hist, image_to_state(state.pob))

			print('  -- Episode %d, action %d, reward %f, (step %d)' %
					(nepisodes, action, state.reward, epi_step))

		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			if state.terminal:
				nepisodes_solved += 1
			# start a new game
			state, state_hist = reset_game()

		if step % opt.prog_freq == 0:
			print("- Step:", step)

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

	# 2. calculate statistics
	print(float(nepisodes_solved) / float(nepisodes))
	# 3. TODO perhaps  do some additional analysis


def predict(agents, input):
	results = []
	for agent in agents:
		results.append(np.argmax(agent.predict(input)))

	print("Predict generated these results:\n{0}".format(results))
	counts = np.bincount(results)
	return np.argmax(counts)

test_agent()
