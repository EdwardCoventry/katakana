import numpy as np

CHAR_CODE_START = 1
CHAR_CODE_PADDING = 0


def build_characters_encoding(names, convert_to_lower):
    """
    :param names: list of strings
    :return: (encoding, decoding, count)
    """
    count = 2
    encoding = {}
    decoding = {1: 'START'}
    for c in {c for name in names for c in name}:
        c = c.lower() if convert_to_lower else c
        encoding[c] = count
        decoding[count] = c
        count += 1
    return encoding, decoding, count


def transform(encoding, data, vector_size, convert_to_lower):
    """
    :param encoding: encoding dict built by build_characters_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size), dtype='int')
    for i, word in enumerate(data):
        for j, c in enumerate(word[:vector_size]):
            c = c.lower() if convert_to_lower else c
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
