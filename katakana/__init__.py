
from . import getconfig, model

loaded_model = None
input_encoding = None
input_decoding = None
output_encoding = None
output_decoding = None
config = None


def load_default_model(version=None):

    if version is None:
        use_model_config = getconfig.get_use_model_config()
        version = use_model_config['version']

    global loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config

    # print('loading model ...')
    loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config = \
        model.load(version=version)
    # print('model loaded ...')


def to_katakana(text, version=None):
    if loaded_model is None:
        load_default_model(version=version)

    if model_config['convert_to_lower']:
        text = text.lower()

    return model.to_katakana(
        text=text,
        model=loaded_model,
        input_encoding=input_encoding,
        output_decoding=output_decoding,
        input_length=model_config['vector_length'],
        output_length=model_config['vector_length'],
        convert_to_lower=model_config['convert_to_lower']
    )
