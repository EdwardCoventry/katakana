import numpy as np
import unidecode

from katakana import encoding


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
