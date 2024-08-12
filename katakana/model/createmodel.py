import platform
from keras.layers import (
    Input, Embedding, Dense, LSTM, Concatenate, Bidirectional,
    LayerNormalization, TimeDistributed)
from keras.models import Model

# Use the legacy Adam optimizer for macOS
if platform.system() == 'Darwin':
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from keras.optimizers import Adam


# tf.config.experimental_run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# Define constants
EMBEDDING_DIM = 128
LSTM_UNITS = 128
BATCH_SIZE = 64  # You can adjust this as necessary based on your hardware capabilities
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

def create_model(input_dict_size, output_dict_size, input_length, output_length):
    # Encoder
    encoder_inputs = Input(shape=(input_length,))
    encoder_embedding = Embedding(input_dim=input_dict_size, output_dim=EMBEDDING_DIM, mask_zero=True)(encoder_inputs)
    encoder_lstm = Bidirectional(
        LSTM(LSTM_UNITS, return_state=True, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(output_length,))
    decoder_embedding = Embedding(input_dim=output_dict_size, output_dim=EMBEDDING_DIM, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(LSTM_UNITS * 2, return_sequences=True, return_state=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)
    decoder_lstm_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Apply LayerNormalization
    decoder_lstm_outputs = LayerNormalization()(decoder_lstm_outputs)

    # Dense layer with TimeDistributed for sequence output
    decoder_dense = TimeDistributed(Dense(output_dict_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_lstm_outputs)

    # Define the model
    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    seq2seq_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    seq2seq_model.summary()

    return seq2seq_model
