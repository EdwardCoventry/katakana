
from katakana.getconfig import get_use_model_config
from katakana.model import load_model

loaded_model = None
input_encoding = None
input_decoding = None
output_encoding = None
output_decoding = None
config = None
use_model_config = None
model_config = None


def load_default_model(version=None, checkpoint=None, use_tflite=True):

    global loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config, use_model_config

    if loaded_model is None:
        if version is None:
            if use_model_config is None:
                use_model_config = get_use_model_config()
            version = use_model_config['version']

        if checkpoint is None:
            if use_model_config is None:
                use_model_config = get_use_model_config()
            checkpoint = use_model_config['checkpoint']

        loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config = \
            load_model(version=version, checkpoint=checkpoint, use_tflite=use_tflite)

    return loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config, use_model_config


