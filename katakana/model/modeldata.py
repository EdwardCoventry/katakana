import numpy as np

from katakana.encoding import encoding


def create_model_data(encoded_input, encoded_output, output_dict_size):
    """
    :param encoded_input: numpy array of encoded input sequences
    :param encoded_output: numpy array of encoded output sequences
    :param output_dict_size: size of the output dictionary
    :return: tuple of numpy arrays (encoder_input, decoder_input, decoder_output)
    """
    encoder_input = encoded_input

    # Adjust decoder_input to include SOS token and shift the sequence to the right
    decoder_input = np.zeros_like(encoded_output)
    decoder_input[:, 0] = encoding.CHAR_CODE_START  # SOS token at the start
    decoder_input[:, 1:] = encoded_output[:, :-1]  # Shift the output by one

    # Convert encoded_output to one-hot encoding, ignoring the last EOS token if necessary
    decoder_output = np.eye(output_dict_size)[encoded_output]

    return encoder_input, decoder_input, decoder_output