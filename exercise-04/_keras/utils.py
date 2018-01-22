import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys


class Options:
	def __init__(self, run_name="", model_name=""):
		# General config
		self.disp_on = True  # you might want to set it to False for speed
		self.vis_model = True  # if true: save model diagram
		self.print_interv = 1000  # Interval for printing statistics
		self.map_ind = 1
		self.change_tgt = False
		self.path_prefix = "data/"
		self.states_fil = self.path_prefix + run_name + "states.csv"
		self.labels_fil = self.path_prefix + run_name + "labels.csv"
		self.network_fil = self.path_prefix + run_name + "network.json"
		self.weights_fil = self.path_prefix + run_name + "network.h5"
		self.model_path = self.path_prefix + "images/{}.png"
		self.acc_reward_train_figure_path = self.path_prefix + "images/" + run_name + "train_acc_reward.png"
		self.acc_reward_test_figure_path = self.path_prefix + "images/" + run_name + "test_acc_reward.png"
		self.log_path = self.path_prefix + 'logs/' + run_name + str(
			datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
		
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
			self.state_siz = (self.pob_siz * self.cub_siz) ** 2  # when use pob as input
		if self.change_tgt:
			self.tgt_y = None
			self.tgt_x = None
		self.act_num = 5
		
		# traing hyper params
		self.hist_len = 4
		self.minibatch_size = 32
		self.eval_nepisodes = 10
		self.discount = 0.99
		self.dual_model = True
		self.prev_exp = 200000  # Let agent move for this number of steps before training
		self.steps = 300000  # Number of total steps
		self.train_interval = 100  # Specifies number of steps between each training round
		self.weigth_sync_interval = 10000  # Number of steps between copying weights to target model
		self.expl_prob = 0.7  # Probability for a random movement
		self.expl_prob_decay = 0.99999
		self.expl_prob_min = 0.2


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


def plot_reward(acc_reward, path):
	plt.figure(2)
	plt.plot(acc_reward)
	plt.title('Training')
	plt.ylabel('Acc. Reward')
	plt.xlabel('Step')
	plt.grid(True)
	plt.savefig(path)


# The following function was taken from https://gist.github.com/aubricus/
# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		bar_length  - Optional  : character length of bar (Int)
	"""
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '█' * filled_length + '-' * (bar_length - filled_length)
	sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix))
	# sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
	
	if iteration == total:
		sys.stdout.write('\n')
		sys.stdout.flush()
