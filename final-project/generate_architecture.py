import xml.etree.ElementTree as ET


activation = ['None', 'relu', 'tanh']
n_filters = [8, 16, 32, 64, 128]
kernel_size = range(2, 9)
# Generates a sequence of integers from 1 to size/2 + 1, where size is the kernel size for which we need to generate strides
strides = lambda size: range(1, size/2 + 2)
padding = 'valid'

layer_params_1 = {
	'n_filters' : str(n_filters[1]), # i.e. 16
	'kernel_size' : str(kernel_size[-1]), # i.e. 8
	'strides' : str(strides(kernel_size[-1])[-2]), # i.e. 4
	'padding' : padding,
	'activation' : activation[2],
}

layer_param_node_1 = ET.Element('LayerParam', attrib=layer_params_1)

layer_params_2 = {
	'n_filters' : str(n_filters[2]), # i.e. 32
	'kernel_size' : str(kernel_size[2]), # i.e. 4
	'strides' : str(strides(kernel_size[2])[1]), # i.e. 2
	'padding' : padding,
	'activation' : activation[2],
}

layer_param_node_2 = ET.Element('LayerParam', attrib=layer_params_2)

layer_params_3 = {
	'n_filters' : str(n_filters[3]), # i.e. 64
	'kernel_size' : str(kernel_size[1]), # i.e. 3
	'strides' : str(strides(kernel_size[1])[0]), # i.e. 1
	'padding' : padding,
	'activation' : activation[2],
}

layer_param_node_3 = ET.Element('LayerParam', attrib=layer_params_3)

layer_params_node = ET.Element('layer_params')
layer_params_node.append(layer_param_node_1)
layer_params_node.append(layer_param_node_2)
layer_params_node.append(layer_param_node_3)
model_params_node = ET.Element('modelParams')
model_params_node.append(layer_params_node)

ET.ElementTree(model_params_node).write('config/architectures/dueling_paper_architecture.cfg')
