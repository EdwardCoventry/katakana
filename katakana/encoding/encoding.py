import numpy as np

LEN_START_AND_END_CODES = 2

class SPECIAL_CODES:
    PAD = 'PAD'
    SOS = 'SOS'
    EOS = 'EOS'


class _SPECIAL_CODES_ENCODING:
    PAD = 0
    SOS = 1
    EOS = 2


def build_characters_encoding(names, config):
    """
    :param names: list of strings
    :return: (encoding, decoding)
    """
    encoding = {
        SPECIAL_CODES.PAD: _SPECIAL_CODES_ENCODING.PAD,
        SPECIAL_CODES.SOS: _SPECIAL_CODES_ENCODING.SOS,
        SPECIAL_CODES.EOS: _SPECIAL_CODES_ENCODING.EOS
    }
    decoding = {
        _SPECIAL_CODES_ENCODING.PAD: SPECIAL_CODES.PAD,
        _SPECIAL_CODES_ENCODING.SOS: SPECIAL_CODES.SOS,
        _SPECIAL_CODES_ENCODING.EOS: SPECIAL_CODES.EOS
    }

    chars = {c for name in names for c in name}

    for i, c in enumerate(chars, start=len(encoding)):
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
    transformed_data = np.full((len(data), vector_length), _SPECIAL_CODES_ENCODING.PAD, dtype='int')

    for i, word in enumerate(data):
        encoded_word = [encoding[SPECIAL_CODES.SOS]] + \
                       [encoding.get(c, _SPECIAL_CODES_ENCODING.PAD) for c in
                        word[:vector_length - LEN_START_AND_END_CODES]] + \
                       [encoding[SPECIAL_CODES.EOS]]  # Adding SOS at the start and EOS at the end

        for j, c in enumerate(encoded_word[:vector_length]):
            transformed_data[i][j] = c

    return transformed_data


def decode(decoding, vector):
    """
    :param decoding: decoding dict built by build_characters_encoding()
    :param vector: an encoded vector
    """
    text = ''
    for x in vector:
        if x == _SPECIAL_CODES_ENCODING.EOS or x == _SPECIAL_CODES_ENCODING.PAD:
            break
        text += decoding.get(x, '')  # Use an empty string for unknown codes
    return text