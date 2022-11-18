
from . import getconfig, model

loaded_model = None
input_encoding = None
input_decoding = None
output_encoding = None
output_decoding = None
config = None


def load_default_model(version=None):

    if version is None:
        general_config = getconfig.get_config()
        version = general_config['version']

    global loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, config

    # print('loading model ...')
    loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, config = \
        model.load(save_dir='trained_models', version=version)
    # print('model loaded ...')


def to_katakana(text, version=None):
    if loaded_model is None:
        load_default_model(version=version)

    return model.to_katakana(
        text=text,
        model=loaded_model,
        input_encoding=input_encoding,
        output_decoding=output_decoding,
        input_length=config['vector_length'],
        output_length=config['vector_length']
    )
