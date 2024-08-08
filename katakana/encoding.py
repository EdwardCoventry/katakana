import numpy as np
from katakana.formattext import format_text

CHAR_CODE_START = 1
CHAR_CODE_PADDING = 0


def build_characters_encoding(names, config):
    """
    :param names: list of strings
    :return: (encoding, decoding, count)
    """
    encoding = {}
    decoding = {1: 'START'}

    chars = {c
             for name in names
             for c in name}

    for i, c in enumerate(chars, start=2):
        encoding[c] = i
        decoding[i] = c
        i += 1
    return encoding, decoding


def transform(encoding, data, config):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    vector_length = config['vector_length']
    transformed_data = np.full((len(data), vector_length), CHAR_CODE_PADDING, dtype='int')
    for i, word in enumerate(data):
        for j, c in enumerate(word[:vector_length]):
            transformed_data[i][j] = encoding[c]
    return transformed_data


def decode(decoding, vector):
    """
    :param decoding: decoding dict built by build_characters_encoding()
    :param vector: an encoded vector
    """
    text = ''
    for x in vector:
        if x == 0:
            break
        text += decoding[x]
    return text
