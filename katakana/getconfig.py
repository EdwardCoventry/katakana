import os.path
import yaml


default_path = os.path.join(__file__, '..')

def get_config(path=default_path, name='config.yaml'):
	# Read YAML file
	with open(os.path.join(path, name), 'r') as stream:
		return yaml.safe_load(stream)

def write_config(config, path, name='config.yaml'):
	# Write YAML file
	with open(os.path.join(path, name), 'w+', encoding='utf8') as outfile:
		yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

def get_training_config(path=default_path):
	return get_config(path, 'trainingconfig.yaml')

def get_use_model_config(path=default_path):
	return get_config(path, 'usemodelconfig.yaml')

def get_model_config(path=default_path):
	return get_config(path, 'modelconfig.yaml')

def write_model_config(config, path, name='modelconfig.yaml'):
	# Write YAML file
	with open(os.path.join(path, name), 'w+', encoding='utf8') as outfile:
		yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)
