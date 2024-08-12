import numpy as np
import unidecode

from katakana import encoding
from katakana.encoding import SPECIAL_CODES


def convert_to_katakana(text, model, input_encoding, output_decoding, config):
    """
    Converts an English text to Katakana using the trained model.

    :param text: The input English text.
    :param model: The trained model for conversion.
    :param input_encoding: The input encoding dictionary.
    :param output_decoding: The output decoding dictionary.
    :param config: Configuration dictionary.
    :return: Converted Katakana text.
    """
    if config['convert_to_unidecode']:
        text = unidecode.unidecode(text)
    if config['convert_to_lower']:
        text = text.lower()

    encoder_input = encoding.transform(input_encoding, [text], config)
    decoder_input = np.zeros(shape=(1, config['vector_length']))
    decoder_input[:, 0] = input_encoding[SPECIAL_CODES.SOS]

    for i in range(1, config['vector_length']):
        output = model.predict([encoder_input, decoder_input], verbose=0).argmax(axis=2)
        next_char = output[0, i - 1]
        if next_char == input_encoding[SPECIAL_CODES.EOS]:
            break
        else:
            decoder_input[:, i] = next_char

    decoder_output = decoder_input
    return encoding.decode(output_decoding, decoder_output[0][1:])