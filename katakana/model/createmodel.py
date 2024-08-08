import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Dropout, LSTM, Masking
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import TimeDistributed

tf.config.experimental_run_functions_eagerly(True)

def create_model(input_dict_size, output_dict_size, input_length, output_length):
    # Encoder
    encoder_inputs = Input(shape=(input_length,))
    encoder_embedding = Embedding(input_dim=input_dict_size, output_dim=64, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(output_length,))
    decoder_embedding = Embedding(input_dim=output_dict_size, output_dim=64, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(output_dict_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return seq2seq_model
