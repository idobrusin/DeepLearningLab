import xml.etree.ElementTree as ET
from numpy import logspace


steps = 50000
prefix = 'dueling'
hlength = [4, 8]
algo = [1, 2, 3, 4]
learning_rate = logspace(
	start=-5, 
	stop=-3,
	num=10
)

decay_func = lambda maxx, minn, steps: float(maxx - minn) / steps

cfg_list = open('config/simParams/{0}_simParams.txt'.format(prefix), 'a')

for hl in hlength:
	for lr in learning_rate:
		sim_value_params = {
			'steps': str(steps),
			'prefix' : prefix,
			'hlength' : str(hl),
			'algo' : str(algo[3]),
			'learning_rate' : str(lr)
		}

		maxx = 0.5
		minn = 0.1
		for frac in [0.1, 0.3, 0.5]: # fraction of total steps to use for decay
			sim_params_node = ET.Element('simParams')
			for key, value in sim_value_params.items():
				sim_params_node.append(ET.Element(key, {'value' : value}))

			exploration_attribs = {
				'max' : str(maxx),
				'min' : str(minn),
				'decay' : str(decay_func(maxx, minn, steps * frac))
			}
			exploration_node = ET.Element('exploration_prob', attrib=exploration_attribs)
			sim_params_node.append(exploration_node)
			cfg_file_name = "sim_params_hl{0}_lr{1:.2E}_frac{2}".format(hl, lr, frac)
			cfg_list.write('{0}\n'.format(cfg_file_name))

			ET.ElementTree(sim_params_node).write('config/simParams/{0}.cfg'.format(cfg_file_name))

cfg_list.close()
