import numpy as np


class Options:
	"""
	General options for the grid world.
	"""
	
	def __init__(self):
		
		# ------------ General config----
		# -------------------------------
		
		# Control whether output should be rendered or not
		self.disp_on = True  # you might want to set it to False for speed
		
		# Index of map to use, taken from maps.py
		self.map_index = 4
		
		# TODO: ???
		# self.change_tgt = True
		self.change_tgt = False
		
		# Location of generated (optimal) states file
		self.states_file = "data/training/states.csv"
		
		# Location of generated (optimal) labels file
		self.labels_file = "data/training/labels.csv"
		
		# Location of network architecture - generated from keras
		self.network_file = "network.json"
		
		# Location of weights file - generated from training the agent
		self.weights_file = "network.h5"
		
		# Folder containing models
		self.model_folder = "data/training/"
		
		# ---------- Simulator config ---
		# -------------------------------
		# Controls frame rate
		self.disp_interval = .005
		
		if self.map_index == 0:
			# size of a cube in the image
			self.cube_size = 5  # size of a cube in the image
			
			# partial observation window
			self.partial_obs_size = 5
			
			# this defines the goal position
			self.target_y = 12
			self.target_x = 11
			self.early_stop = 50
		
		elif self.map_index == 1:
			# size of a cube in the image
			self.cube_size = 10
			
			# partial observation window
			self.partial_obs_size = 3  # for partial observation
			
			# this defines the goal position
			self.target_y = 5
			self.target_x = 5
			self.early_stop = 75
		elif self.map_index == 2:
			# size of a cube in the image
			self.cube_size = 5
			
			# partial observation window
			self.partial_obs_size = 5  # for partial observation
			
			# this defines the goal position
			self.target_y = 12
			self.target_x = 16
			self.early_stop = 50
		elif self.map_index == 3:
			# size of a cube in the image
			self.cube_size = 5
			
			# partial observation window
			self.partial_obs_size = 5  # for partial observation
			
			# this defines the goal position
			self.target_y = 15
			self.target_x = 11
			self.early_stop = 50
		elif self.map_index == 4:
			# size of a cube in the image
			self.cube_size = 5  # size of a cube in the image
			
			# partial observation window
			self.partial_obs_size = 7
			
			# this defines the goal position
			self.target_y = 13
			self.target_x = 12
			self.early_stop = 50
		else:
			print("Invalid map index: ", self.map_index)
			print("Please use a valid map index, defined in maps.py")
		
		# when use partial_observation as input
		self.state_size = (self.partial_obs_size * self.cube_size) ** 2
		
		if self.change_tgt:
			self.target_x = None
			self.target_y = None
		# TODO: ???
		self.act_num = 5
		
		# traing hyper params
		self.history_length = 4
		self.minibatch_size = 32
		self.n_minibatches = 500
		self.valid_size = 500
		self.eval_n_episodes = 100
		
		self.data_steps = self.n_minibatches * self.minibatch_size + self.valid_size
		self.eval_steps = self.early_stop * self.eval_n_episodes
		self.eval_freq = self.n_minibatches  # evaluate after each epoch
		self.prog_freq = 500  # used for printing every 500 episodes
		
		self.num_models = 5
	
	#	self.num_models = None


class State:
	"""
	return tuples made easy
	"""
	
	def __init__(self, action, reward, screen, terminal, pob):
		self.action = action
		self.reward = reward
		self.screen = screen
		self.terminal = terminal
		self.pob = pob


# The following functions were taken from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
def rgb2gray(rgb):
	if rgb.ndim == 2:
		return np.ascontiguousarray(rgb)
	
	gray = 0.2125 * rgb[..., 0]
	gray[:] += 0.7154 * rgb[..., 1]
	gray[:] += 0.0721 * rgb[..., 2]
	
	return gray
