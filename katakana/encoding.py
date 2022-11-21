import numpy as np

from katakana.formattext import format_text

CHAR_CODE_START = 1
CHAR_CODE_PADDING = 0


def format_char(c, convert_to_lower, convert_to_unidecode):
    return (format_text(c, convert_to_lower, convert_to_unidecode) or c)[0]


def build_characters_encoding(names, config):
    """
    :param names: list of strings
    :return: (encoding, decoding, count)
    """

    encoding = {}
    decoding = {1: 'START'}
    for i, c in enumerate(
            {c
             for name in names
             for c in name
             }, start=2):
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

    transformed_data = np.zeros(shape=(len(data), config['vector_length']), dtype='int')
    for i, word in enumerate(data):
        for j, c in enumerate(word[:config['vector_length']]):
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
