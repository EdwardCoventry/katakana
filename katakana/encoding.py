import numpy as np

CHAR_CODE_START = 1
CHAR_CODE_PADDING = 0


def build_characters_encoding(names, config):
    """
    :param names: list of strings
    :return: (encoding, decoding, count)
    """
    count = 2
    encoding = {}
    decoding = {1: 'START'}
    for c in {c for name in names for c in name}:
        if config['convert_to_unidecode']:
            c = c.lower()
        if config['convert_to_lower']:
            c = c.lower()
        encoding[c] = count
        decoding[count] = c
        count += 1
    return encoding, decoding, count


def transform(encoding, data, config):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), config['vector_length']), dtype='int')
    for i, word in enumerate(data):
        for j, c in enumerate(word[:config['vector_length']]):
            if config['convert_to_unidecode']:
                c = c.lower()
            if config['convert_to_lower']:
                c = c.lower()
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
