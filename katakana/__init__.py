import unidecode

from . import formattext, getconfig, model

loaded_model = None
input_encoding = None
input_decoding = None
output_encoding = None
output_decoding = None
config = None
use_model_config = None


def load_default_model(version=None, checkpoint=None):
    global loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config, use_model_config

    if version is None:
        if use_model_config is None:
            use_model_config = getconfig.get_use_model_config()
        version = use_model_config['version']

    if checkpoint is None:
        if use_model_config is None:
            use_model_config = getconfig.get_use_model_config()
        checkpoint = use_model_config['checkpoint']

    loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config = \
        model.load(version=version, checkpoint=checkpoint)


def to_katakana(text, version=None, checkpoint=None):
    if loaded_model is None:
        load_default_model(version=version, checkpoint=checkpoint)

    text = formattext.format_text(text, model_config['convert_to_lower'], model_config['convert_to_unidecode'])

    return model.to_katakana(
        text=text,
        model=loaded_model,
        input_encoding=input_encoding,
        output_decoding=output_decoding,
        config=model_config
    )
