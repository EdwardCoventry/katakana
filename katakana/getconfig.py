import os.path
import yaml


def get_config(path='.', name='config.yaml'):
	# Read YAML file
	with open(os.path.join('katakana', path, name), 'r') as stream:
		return yaml.safe_load(stream)

def write_config(config, path, name='config.yaml'):
	# Write YAML file
	with open(os.path.join('katakana', path, name), 'w', encoding='utf8') as outfile:
		yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)
