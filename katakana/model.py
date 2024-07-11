import json
import os
import pathlib

import numpy as np
import unidecode
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.models import Model, load_model

from . import encoding, getconfig, converttotflite


def load_config(version=None):
    get_path = lambda filename: pathlib.Path(__file__).parent.joinpath('trained_models', str(version), filename)

    # Read YAML file
    config = getconfig.get_model_config(get_path(''))

    return config


def load(version=None, checkpoint=None, from_path=None, use_tflite=False):
    assert not (checkpoint and from_path), "either pass the checkpoint or the path"

    get_path = lambda filename: pathlib.Path(__file__).parent.joinpath('trained_models', str(version), filename)
    tflite_path = lambda filename: pathlib.Path(__file__).parent.joinpath('trained_models', str(version), 'tflite',
                                                                          filename)

    config = load_config(version)

    input_encoding = json.load(open(get_path('input_encoding.json')))
    input_decoding = json.load(open(get_path('input_decoding.json')))
    input_decoding = {int(k): v for k, v in input_decoding.items()}

    output_encoding = json.load(open(get_path('output_encoding.json')))
    output_decoding = json.load(open(get_path('output_decoding.json')))
    output_decoding = {int(k): v for k, v in output_decoding.items()}

    if from_path:
        model_path = from_path
    elif checkpoint:
        checkpoint_path = get_path('checkpoints')
        checkpoint = str(checkpoint).zfill(2)
        model_path = max(checkpoint_path.glob(f"{checkpoint}-*.{config['file_type']}"),
                         key=lambda path: os.path.getmtime(path))
    else:
        model_path = get_path(f"model.{config['file_type']}")

    if use_tflite:
        tflite_model_path = tflite_path(f"{model_path.stem}.tflite")
        if not tflite_model_path.exists():
            converttotflite.convert_to_tflite(load_model(model_path), tflite_model_path)
        model = converttotflite.TFLiteModelWrapper(str(tflite_model_path))
    else:
        model = load_model(model_path)

    return model, input_encoding, input_decoding, output_encoding, output_decoding, config


def save_config(config):
    """  save a copy of the config file  """
    version_dir = os.path.join('trained_models', str(config['version']))
    get_path = lambda filename: os.path.join(__file__, '..', version_dir, filename)
    getconfig.write_model_config(config, get_path(''))


def save_encodings(input_encoding, input_decoding, output_encoding, output_decoding, config):
    version_dir = os.path.join('trained_models', str(config['version']))
    get_path = lambda filename: os.path.join(__file__, '..', version_dir, filename)

    with open(get_path('input_encoding.json'), 'w') as f:
        json.dump(input_encoding, f)

    with open(get_path('input_decoding.json'), 'w') as f:
        json.dump(input_decoding, f)

    with open(get_path('output_encoding.json'), 'w') as f:
        json.dump(output_encoding, f)

    with open(get_path('output_decoding.json'), 'w') as f:
        json.dump(output_decoding, f)


def save_model(model, config):
    version_dir = os.path.join('trained_models', str(config['version']))
    get_path = lambda filename: os.path.join(__file__, '..', version_dir, filename)

    model.save(get_path(f"model.{config['file_type']}"))


def create_model(
        input_dict_size,
        output_dict_size,
        input_length,
        output_length):
    encoder_input = Input(shape=(input_length,))
    decoder_input = Input(shape=(output_length,))

    encoder = Embedding(input_dict_size, 64, input_length=input_length, mask_zero=True)(encoder_input)
    encoder = LSTM(64, return_sequences=False)(encoder)

    decoder = Embedding(output_dict_size, 64, input_length=output_length, mask_zero=True)(decoder_input)
    decoder = LSTM(64, return_sequences=True)(decoder, initial_state=[encoder, encoder])
    decoder = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


def create_model_data(
        encoded_input,
        encoded_output,
        output_dict_size):
    encoder_input = encoded_input

    decoder_input = np.zeros_like(encoded_output)
    decoder_input[:, 1:] = encoded_output[:, :-1]
    decoder_input[:, 0] = encoding.CHAR_CODE_START

    decoder_output = np.eye(output_dict_size)[encoded_output.astype('int')]

    return encoder_input, decoder_input, decoder_output


# =====================================================================


def to_katakana(text, model, input_encoding, output_decoding, config):
    if config['convert_to_unidecode']:
        text = unidecode.unidecode(text)
    if config['convert_to_lower']:
        text = text.lower()

    encoder_input = encoding.transform(input_encoding, [text], config)
    decoder_input = np.zeros(shape=(len(encoder_input), config['vector_length']))
    decoder_input[:, 0] = encoding.CHAR_CODE_START
    for i in range(1, config['vector_length']):
        output = model.predict([encoder_input, decoder_input], verbose=0).argmax(axis=2)
        decoder_input[:, i] = output[:, i]

    decoder_output = decoder_input
    return encoding.decode(output_decoding, decoder_output[0][1:])
