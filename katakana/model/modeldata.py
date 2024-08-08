import numpy as np

from encoding import encoding


def create_model_data(encoded_input, encoded_output, output_dict_size):
    encoder_input = encoded_input

    # Initialize decoder_input with zeros and shift encoded_output
    decoder_input = np.zeros_like(encoded_output)
    decoder_input[:, 1:] = encoded_output[:, :-1]
    decoder_input[:, 0] = encoding.CHAR_CODE_START

    # Convert encoded_output to one-hot encoding
    decoder_output = np.eye(output_dict_size)[encoded_output]

    return encoder_input, decoder_input, decoder_output
