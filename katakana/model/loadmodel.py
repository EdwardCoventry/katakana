import json
import os
import pathlib
from keras.models import load_model
from katakana import converttotflite, getconfig

def get_model_path(version, filename):
    return pathlib.Path(__file__).parent.parent / 'trained_models' / str(version) / filename

def get_tflite_path(version, filename):
    return pathlib.Path(__file__).parent.parent / 'trained_models' / str(version) / 'tflite' / filename

def load(version=None, checkpoint=None, from_path=None, use_tflite=False):
    assert not (checkpoint and from_path), "Either pass the checkpoint or the path, not both."

    config = load_config(version)

    input_encoding_path = get_model_path(version, 'input_encoding.json')
    input_decoding_path = get_model_path(version, 'input_decoding.json')
    output_encoding_path = get_model_path(version, 'output_encoding.json')
    output_decoding_path = get_model_path(version, 'output_decoding.json')

    input_encoding = json.load(open(input_encoding_path))
    input_decoding = {int(k): v for k, v in json.load(open(input_decoding_path)).items()}
    output_encoding = json.load(open(output_encoding_path))
    output_decoding = {int(k): v for k, v in json.load(open(output_decoding_path)).items()}

    if from_path:
        model_path = from_path
    elif checkpoint:
        checkpoint_path = get_model_path(version, 'checkpoints')
        checkpoint = str(checkpoint).zfill(2)
        model_path = max(checkpoint_path.glob(f"{checkpoint}-*.{config['file_type']}"),
                         key=os.path.getmtime)
    else:
        model_path = get_model_path(version, f"model.{config['file_type']}")

    if use_tflite:
        tflite_model_path = get_tflite_path(version, f"{model_path.stem}.tflite")
        if not tflite_model_path.exists():
            converttotflite.convert_to_tflite(load_model(model_path), tflite_model_path)
        model = converttotflite.TFLiteModelWrapper(str(tflite_model_path))
    else:
        model = load_model(model_path)

    return model, input_encoding, input_decoding, output_encoding, output_decoding, config

def load_config(version=None):
    config_path = get_model_path(version, '')
    return getconfig.get_model_config(config_path)
