import tensorflow as tf
from keras.layers import (Input, Embedding, Dense, Dropout, LSTM, Concatenate, Bidirectional, LayerNormalization,
                          Attention, MultiHeadAttention)
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import TimeDistributed, Layer

# this is really slow but at least it runs...
# tf.config.experimental_run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# Define constants
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 64  # You can adjust this as necessary based on your hardware capabilities

def create_model(input_dict_size, output_dict_size, input_length, output_length):
    # Encoder
    encoder_inputs = Input(shape=(input_length,))
    encoder_embedding = Embedding(input_dim=input_dict_size, output_dim=EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
    encoder_lstm_1 = Bidirectional(
        LSTM(LSTM_UNITS, return_state=True, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    encoder_lstm_2 = Bidirectional(
        LSTM(LSTM_UNITS, return_state=True, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))

    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm_1(encoder_embedding)
    encoder_outputs, forward_h2, forward_c2, backward_h2, backward_c2 = encoder_lstm_2(encoder_outputs)

    state_h = Concatenate()([forward_h2, backward_h2])
    state_c = Concatenate()([forward_c2, backward_c2])
    encoder_states = [state_h, state_c]
    encoder_outputs = LayerNormalization()(encoder_outputs)

    # Decoder
    decoder_inputs = Input(shape=(output_length,))
    decoder_embedding = Embedding(input_dim=output_dict_size, output_dim=EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3)
    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_lstm_outputs = LayerNormalization()(decoder_lstm_outputs)

    # Attention Layer
    attention = Attention()([decoder_lstm_outputs, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([decoder_lstm_outputs, attention])

    # Dense layer
    decoder_dense = TimeDistributed(Dense(output_dict_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_combined_context)

    # Define the model
    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    seq2seq_model.summary()

    return seq2seq_model