import numpy as np


class Options:
	def __init__(self):
		self.disp_on = True  # you might want to set it to False for speed
		self.map_ind = 1
		self.change_tgt = False
		self.data_path_prefix = "data/"
		self.log_path_prefix = "logs/"
		self.model_path_prefix = "models/"
		self.model_diagram_path_prefix = "images/"
		self.model_file = "_run"
		self.model_diagram_file = "_arch.png"
		self.acc_rew_train_file = "_train_acc_reward.png"
		self.acc_rew_test_file = "_test_acc_reward.png"
		
		# simulator config
		self.disp_interval = .005
		
		if self.map_ind == 0:
			self.cub_siz = 5
			self.pob_siz = 5  # for partial observation
			# this defines the goal position
			self.tgt_y = 12
			self.tgt_x = 11
			self.early_stop = 50
		elif self.map_ind == 1:
			self.cub_siz = 10
			self.pob_siz = 3  # for partial observation
			# this defines the goal position
			self.tgt_y = 5
			self.tgt_x = 5
			self.early_stop = 75
			# when use pob as input
			self.state_siz = (self.pob_siz * self.cub_siz) ** 2
		
		if self.change_tgt:
			self.tgt_y = None
			self.tgt_x = None
		self.act_num = 5
		
		# traing hyper params
		self.hist_len = 4
		self.minibatch_size = 32
		
		self.eval_nepisodes = 10
		self.exploration_prob = 0.7
		self.exploration_prob_min = 0.2
		self.exploration_decay = 0.9999  # will decay exploration probability: e.g. from 0.7 to 0.2 in 12000 steps
		self.prior_experience = 1000
		self.steps = 20000
		self.target_update_interval = 1000


class State:  # return tuples made easy
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
