import json
import os
import shutil

import numpy as np
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
# from tensorflow.keras.models import Model, load_model

from keras.models import Model, load_model

from . import encoding, config

DEFAULT_INPUT_LENGTH = config.DEFAULT_VECTOR_LENGTH
DEFAULT_OUTPUT_LENGTH = config.DEFAULT_VECTOR_LENGTH


def load(save_dir='trained_models', version=None):

    version_dir = os.path.join(save_dir, version)
    get_path = lambda filename: os.path.join(version_dir, filename)

    input_encoding = json.load(open(get_path('input_encoding.json')))
    input_decoding = json.load(open(get_path('input_decoding.json')))
    input_decoding = {int(k): v for k, v in input_decoding.items()}

    output_encoding = json.load(open(get_path('output_encoding.json')))
    output_decoding = json.load(open(get_path('output_decoding.json')))
    output_decoding = {int(k): v for k, v in output_decoding.items()}

    model = load_model(get_path('model.h5'))
    return model, input_encoding, input_decoding, output_encoding, output_decoding


def save(model, input_encoding, input_decoding, output_encoding, output_decoding,
         save_dir='trained_models', version=None):

    version_dir = os.path.join(save_dir, version)
    get_path = lambda filename: os.path.join(version_dir, filename)

    if os.path.exists(version_dir):
        shutil.rmtree(version_dir)

    os.mkdir(version_dir)

    with open(get_path('input_encoding.json'), 'w') as f:
        json.dump(input_encoding, f)

    with open(get_path('input_decoding.json'), 'w') as f:
        json.dump(input_decoding, f)

    with open(get_path('output_encoding.json'), 'w') as f:
        json.dump(output_encoding, f)

    with open(get_path('output_decoding.json'), 'w') as f:
        json.dump(output_decoding, f)

    model.save(get_path('model.h5'))


def create_model(
        input_dict_size,
        output_dict_size,
        input_length=DEFAULT_INPUT_LENGTH,
        output_length=DEFAULT_OUTPUT_LENGTH):

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


def to_katakana(text, model, input_encoding, output_decoding,
                input_length=DEFAULT_INPUT_LENGTH,
                output_length=DEFAULT_OUTPUT_LENGTH):

    encoder_input = encoding.transform(input_encoding, [text.lower()], input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), output_length))
    decoder_input[:, 0] = encoding.CHAR_CODE_START
    for i in range(1, output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:, i] = output[:, i]

    decoder_output = decoder_input
    return encoding.decode(output_decoding, decoder_output[0][1:])