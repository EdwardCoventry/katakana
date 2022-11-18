import os.path
import yaml


root_path = os.path.join(__file__, '..')

def get_config(path='.', name='config.yaml'):
	"""  Read YAML file """
	with open(os.path.join(root_path, path, name), 'r') as stream:
		return yaml.safe_load(stream)

def write_config(config, path, name='config.yaml'):
	"""  Write YAML file  """
	with open(os.path.join(root_path, path, name), 'w+', encoding='utf8') as outfile:
		yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

def get_training_config(path='.'):
	return get_config(path, 'trainingconfig.yaml')

def get_use_model_config(path='.'):
	return get_config(path, 'usemodelconfig.yaml')

def get_model_config(path='.'):
	return get_config(path, 'modelconfig.yaml')

def write_model_config(config, path):
	write_config(config, path, name='modelconfig.yaml')
