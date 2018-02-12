import xml.etree.ElementTree as ET


model_params_file = ('dueling_paper_architecture')
with open("config/simParams/dueling_simParams.txt", 'r') as sim_params_list:
	ct = 0
	for line in sim_params_list:
		root = ET.Element('config')
		model_params_node = ET.Element('modelParams', {"source" : model_params_file})
		root.append(model_params_node)
		sim_params_node = ET.Element('simParams', {"source" : line.strip()})
		root.append(sim_params_node)
		ET.ElementTree(root).write('config/dueling_paper_arch_sim_{0}.cfg'.format(ct))
		ct += 1
