import numpy as np
import xml.etree.ElementTree as ET
from model_utils import *
from enum import Enum


class Options:
	def __init__(self):
		#~~!!CONSTANTS!!~~
		self.OUTPUT_TENSOR_NAME = "output"
		self.ONE_HOT_ACTION_TENSOR_NAME = "one_hot_action"
		self.LOSS_TENSOR_NAME = "loss"
		self.SUMMARY_LOSS_NAME = "Loss"
		self.SUMMARY_NUM_FINISHED_EP = "Total Finished episode count"
		self.SUMMARY_ACC_REWARD = "Accumulated reward"
		self.SUMMARY_STEP_PER_EPISODE = "Steps/FinishedEp"
		self.SUMMARY_STEP_PER_EPISODE_AVG = "AvgSteps/FinishedEp"
		self.supported_mtypes = ['Conv', 'ConvPool', 'Dense']
		self.supported_rtypes = ['TRAIN', 'TEST']

		#~~!!INTERNAL SETTINGS!!~~
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
		self.cfg_dir = "config/"
		self.cfg_prefix = ""
		self.default_config_file = "default"
		self._config_file_suffix = ".cfg"
		self.mtype = 'Conv'	# Available options: Conv, ConvPool, Dense
		self.rtype = None
		self.prefix = None

		#~~!!GLOBAL SHARED OBJECTS!!~~
		self.summary_writer = None
		self.training = False
		self.testing = False
		self.chosen_algo = None
		self._model_params = None
		
		# simulator config
		self.disp_interval = .005
		self.maxlen = 100000
		
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
		self.learning_rate = 1e-4
		self.eval_nepisodes = 10
		self.exploration_prob = 0.7
		self.exploration_prob_min = 0.2
		self.steps = 20000
		self.exploration_decay = (self.exploration_prob - self.exploration_prob_min) / self.steps  # will decay exploration probability: e.g. from 0.7 to 0.2 in 12000 steps
		self.prior_experience = 1000
		self.steps = 20000
		self.target_update_interval = 1000

	def generate_model_name(self):
		return '_'.join([self.prefix, self.cfg_prefix])

	def generate_run_name(self):
		return '_'.join([self.prefix, self.cfg_prefix])

	def get_model_params(self):
		return self._model_params

	def set_algo(self, choice):
		self.chosen_algo = DqnAlgorithm(choice)

	def load_config_file(self, cfg_file_name):
		self.cfg_prefix = cfg_file_name
		cfg_file_path = self.cfg_dir + cfg_file_name + self._config_file_suffix
		tree = ET.parse(cfg_file_path)
		root = tree.getroot()
		for child in root:
			if child.tag == 'modelParams':
				self._read_model_params(child)
			elif child.tag == 'simParams':
				self._read_sim_params(child)
			else:
				print("Unhandled element in config file with tag {0} and attributes: {1}".format(child.tag, child.attrib))
		return

	def _read_value_of_xml_element(self, root):
		if 'value' in root.attrib:
			return root.attrib['value']
		else:
			raise Exception('The element with tag {0} must have an attribute "value".'.format(root.tag))
		
	def _read_optimizer(self, root):
		return self._read_value_of_xml_element(root)

	def _read_loss_function(self, root):
		return self._read_value_of_xml_element(root)

	def _read_metrics(self, root):
		metrics = []
		for child in root:
			if child.tag != 'metric':
			 	raise Exception('Encountered unknown child tag in metrics tag: {0}'.format(child.tag))
			metrics.append(child.tag)
		return metrics
	
	def _read_dropout_rate(self, root):
		return int(self._read_value_of_xml_element(root))

	def _read_convLayerParams(self, root):
		return ConvLayerParams(**root.attrib)

	def _read_denseLayerParams(self, root):
		print('[FATAL] Reading dense layer model architectures has not been implemented yet.')
		exit(-1)

	_layer_params_handlers = {
		'convLayer': _read_convLayerParams,
		'denseLayer': _read_denseLayerParams
	}

	def _read_layer_params(self, root):
		params = []
		if 'type' in root.attrib:
			reader = self._layer_params_handlers[root.attrib['type']]
		else:
			# By default, we assume a CNN architecture.
			reader = self._read_convLayerParams

		for child in root:
			if child.tag != 'LayerParam':
				raise Exception('Expected node of type LayerParam, received node {0}'.format(child.tag))
			params.append(reader(child))
		return params

	def _read_model_params(self, root):
		#The model params may be read from another file specified in the 'source' attribute of the model_params tag
		model_params_args = {}
		if 'source' in root.attrib:
			filename = root.attrib['source']
			source_file_path = 'config/architectures/' + filename  + '.cfg'
			root = ET.parse(source_file_path).getroot()
			if root.tag != 'modelParams':
				raise Exception('Expected file with modelParams as root node, received file with root node {0} in file {1}'.format(root.tag, source_file_path))
			'''
			import re
			try:
				res = re.search(r'/(.+?)\..+?$', source_file_path).group(1)
			except AttributeError:
				print('[WARN] Could not resolve file name of "source" attribute in model parameters. Using default.') 
				res = 'Source'
			'''

			res = filename
			if self.prefix:
				self.prefix += '_modelFrom_' + res
			else:
				self.prefix = '_modelFrom_' + res

		_model_params_handlers = {
		#	"optimizer" : self._read_optimizer, 
		#	"loss_function" : self._read_loss_function, 
		#	"metrics" : self._read_metrics, 
		#	"dropout_rate" : self._read_dropout_rate, 
			"layer_params" : self._read_layer_params
		}

		for child in root:
			if child.tag in _model_params_handlers:
				model_params_args[child.tag] = _model_params_handlers[child.tag](child)
			else:
				raise Exception("Encountered unknown model parameter tag {0}".format(child.tag))
		# Create an object of class ModelParams using the generated dictionary of key-word arguments and store it for internal usage
		self._model_params = ModelParams(**model_params_args)
		print("[INFO] Finished reading model parameters from file.")
		return

	def _read_steps(self, root):
		self.steps = int(self._read_value_of_xml_element(root))
		return

	def _read_prefix(self, root):
		prefix = self._read_value_of_xml_element(root)
		# TODO: Currently, the when reading from source config files, the latter part of the prefix is generated before the actual prefix itself. In the future, this should be rectified and made less ugly.
		if self.prefix:
			self.prefix = prefix + self.prefix
		else:
			self.prefix = prefix
		return

	def _read_rtype(self, root):
		rtype = self._read_value_of_xml_element(root)
		if rtype in self.supported_rtypes:
			self.rtype = rtype
		else:
			raise Exception("Given rtype {0} is not supported. Must be one of {1}".format(rtype, self.supported_rtypes))
		return

	def _read_hlength(self, root):
		self.hist_len = int(self._read_value_of_xml_element(root))
		return

	def _read_algo(self, root):
		self.set_algo(int(self._read_value_of_xml_element(root)))
		return

	def _read_lrate(self, root):
		self.learning_rate = float(self._read_value_of_xml_element(root))
		return

	def _read_exploration(self, root):
		def _set_max_explor(value):
			self.exploration_prob = value
			return

		def _set_min_explor(value):
			self.exploration_prob_min = value
			return

		def _set_explor_decay(value):
			self.exploration_prob_decay = value
			return

		_funcs = {
			"max" : _set_max_explor,
			"min" : _set_min_explor,
			"decay" : _set_explor_decay,
		}

		for key, value in root.attrib.items():
			try:
				_funcs[key](float(value))
			except Exception as e:
				print("[FATAL] Error while reading key {0} with value {1} from tag {2}".format(key, value, root.tag))
				raise e

	def _read_sim_params(self, root):
		# The sim params may be read from another file specified in the 'source' attribute of the sim_params tag (full path)
		if 'source' in root.attrib:
			filename = root.attrib['source']
			source_file_path = 'config/simParams/' + filename + '.cfg'
			root = ET.parse(source_file_path).getroot()
			if root.tag != 'simParams':
				raise Exception('Expected file with simParams as root node, received file with root node {0} in file {1}'.format(root.tag, source_file_path))
			'''
			import re
			try:
				res = re.search(r'/(.+?)\..+?$', source_file_path).group(1)
			except AttributeError:
				print('[WARN] Could not resolve file name of "source" attribute in sim parameters. Using default.') 
				res = 'Source'
			'''
			res = filename
			if self.prefix:
				self.prefix += '_simFrom_' + res
			else:
				self.prefix = '_simFrom_' + res

		_sim_params_handlers = {
			"steps" : self._read_steps,
			"prefix" : self._read_prefix,
			"rtype" : self._read_rtype,
			"hlength" : self._read_hlength,
			"algo" : self._read_algo,
			"learning_rate" : self._read_lrate,
			"exploration_prob" : self._read_exploration
		}

		for child in root:
			if child.tag in _sim_params_handlers:
				_sim_params_handlers[child.tag](child)
			else:
				raise Exception("Encountered unknown sim paramter tag {0}".format(child.tag))
		print("[INFO] Finished reading simulation parameters from file.")
		return


class DqnAlgorithm(Enum):
	DQN = 1
	DQN_TARGET = 2
	DDQN = 3
	DQN_DUELLING = 4

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


