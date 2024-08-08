import numpy as np

CHAR_CODE_START = 1
CHAR_CODE_PADDING = 0

def build_characters_encoding(names, config):
    """
    :param names: list of strings
    :return: (encoding, decoding, count)
    """
    encoding = { 'PAD': CHAR_CODE_PADDING }  # Define padding explicitly
    decoding = { CHAR_CODE_START: 'START', CHAR_CODE_PADDING: 'PAD' }  # Include padding in decoding

    chars = {c for name in names for c in name}

    for i, c in enumerate(chars, start=2):
        encoding[c] = i
        decoding[i] = c
    return encoding, decoding


def transform(encoding, data, config):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param config: dictionary with configuration, including 'vector_length'
    """
    vector_length = config['vector_length']
    transformed_data = np.full((len(data), vector_length), CHAR_CODE_PADDING, dtype='int')
    for i, word in enumerate(data):
        for j, c in enumerate(word[:vector_length]):
            transformed_data[i][j] = encoding.get(c, CHAR_CODE_PADDING)  # Use padding for unknown characters
    return transformed_data


def decode(decoding, vector):
    """
    :param decoding: decoding dict built by build_characters_encoding()
    :param vector: an encoded vector
    """
    text = ''
    for x in vector:
        if x == CHAR_CODE_PADDING:
            break
        text += decoding.get(x, '')  # Use an empty string for unknown codes
    return text
