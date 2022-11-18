from __future__ import print_function

import shutil
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os.path

from katakana import getconfig, model, encoding, loadcsvdata
import keras.callbacks

training_config = getconfig.get_training_config()

# Load and shuffle  ----------------------

data = loadcsvdata.load_csvs()
data = data.sample(frac=1, random_state=0)

data_input = data['english']
data_output = data['katakana']

data_size = len(data)

train_split_index = int(data_size*90/100)

training_input = data_input[:train_split_index]
training_output = data_output[:train_split_index]
validation_input = data_input[train_split_index:]
validation_output = data_output[train_split_index:]

# Encoding the dataset ----------------------

input_encoding, input_decoding, input_dict_size = encoding.build_characters_encoding(data_input, training_config['convert_to_lower'])
output_encoding, output_decoding, output_dict_size = encoding.build_characters_encoding(data_output, training_config['convert_to_lower'])

encoded_training_input = encoding.transform(input_encoding, training_input, vector_size=training_config['vector_length'], convert_to_lower=training_config['convert_to_lower'])
encoded_training_output = encoding.transform(output_encoding, training_output, vector_size=training_config['vector_length'], convert_to_lower=training_config['convert_to_lower'])
encoded_validation_input = encoding.transform(input_encoding, validation_input, vector_size=training_config['vector_length'], convert_to_lower=training_config['convert_to_lower'])
encoded_validation_output = encoding.transform(output_encoding, validation_output, vector_size=training_config['vector_length'], convert_to_lower=training_config['convert_to_lower'])

# Building the model ----------------------

training_encoder_input, training_decoder_input, training_decoder_output = \
    model.create_model_data(encoded_training_input, encoded_training_output, output_dict_size)

validation_encoder_input, validation_decoder_input, validation_decoder_output = \
    model.create_model_data(encoded_validation_input, encoded_validation_output, output_dict_size)

"""  delete folder if it exists, and (re)make it 
     also make checkpoints folder  """
version_dir = os.path.join('trained_models', training_config['version'])
if os.path.exists(version_dir):
    shutil.rmtree(version_dir)
os.mkdir(version_dir)
checkpoints_dir = os.path.join(version_dir, 'checkpoints')
os.mkdir(checkpoints_dir)

# Building the model ----------------------
seq2seq_model = model.create_model(
    input_dict_size=input_dict_size,
    output_dict_size=output_dict_size,
    input_length=training_config['vector_length'],
    output_length=training_config['vector_length'])

"""  stop when ceases to improve  """
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=3,
                                                        restore_best_weights=True
                                                        )
"""  save all checkpoints  """
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoints_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False,
    verbose=1)

seq2seq_model.fit(
    x=[training_encoder_input, training_decoder_input],
    y=[training_decoder_output],
    validation_data=(
        [validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
    verbose=2,
    batch_size=64,
    epochs=2,
    callbacks=[model_checkpoint_callback, early_stopping_callback])

model.save(
    model=seq2seq_model,
    input_encoding=input_encoding,
    input_decoding=input_decoding,
    output_encoding=output_encoding,
    output_decoding=output_decoding,
    config=training_config)