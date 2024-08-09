from katakana import getconfig, model
from encoding import formattext
import re

loaded_model = None
input_encoding = None
input_decoding = None
output_encoding = None
output_decoding = None
config = None
use_model_config = None


def load_default_model(version=None, checkpoint=None, use_tflite=True):
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
        model.load(version=version, checkpoint=checkpoint, use_tflite=use_tflite)


def to_katakana(text, version=None, checkpoint=None, use_tflite=True):
    if loaded_model is None:
        load_default_model(version=version, checkpoint=checkpoint, use_tflite=use_tflite)

    # Define the pattern to split text at spaces and interpuncts
    split_pattern = re.compile(r'[\s・]+')

    # Split the text
    words = split_pattern.split(text)

    # Process each word separately
    converted_words = []
    for word in words:
        formatted_word = formattext.format_text(word, model_config['convert_to_lower'], model_config['convert_to_unidecode'])
        converted_word = model.to_katakana(
            text=formatted_word,
            model=loaded_model,
            input_encoding=input_encoding,
            output_decoding=output_decoding,
            config=model_config
        )
        converted_words.append(converted_word)

    # Join the converted words back together with a space or interpunct
    joined_text = '・'.join(converted_words)

    return joined_text