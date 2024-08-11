import json
from pathlib import Path

from katakana import getconfig


def save_config(config):
    version_dir = Path(__file__).parent.parent / 'trained_models' / str(config['version'])
    getconfig.write_model_config(config, version_dir)


def save_encodings(input_encoding, input_decoding, output_encoding, output_decoding, config):
    version_dir = Path(__file__).parent.parent / 'trained_models' / str(config['version'])
    get_path = lambda filename: version_dir / filename

    with open(get_path('input_encoding.json'), 'w') as f:
        json.dump(input_encoding, f)

    with open(get_path('input_decoding.json'), 'w') as f:
        json.dump(input_decoding, f)

    with open(get_path('output_encoding.json'), 'w') as f:
        json.dump(output_encoding, f)

    with open(get_path('output_decoding.json'), 'w') as f:
        json.dump(output_decoding, f)


def save_model(model, config):
    version_dir = Path(__file__).parent.parent / 'trained_models' / str(config['version'])
    model.save(version_dir / f"model.{config['file_type']}")