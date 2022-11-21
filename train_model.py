from __future__ import print_function

import glob
import pathlib
import re
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

input_encoding, input_decoding, input_dict_size = encoding.build_characters_encoding(data_input, training_config)
output_encoding, output_decoding, output_dict_size = encoding.build_characters_encoding(data_output, training_config)

encoded_training_input = encoding.transform(input_encoding, training_input, training_config)
encoded_training_output = encoding.transform(output_encoding, training_output, training_config)
encoded_validation_input = encoding.transform(input_encoding, validation_input, training_config)
encoded_validation_output = encoding.transform(output_encoding, validation_output, training_config)

# Building the model ----------------------

training_encoder_input, training_decoder_input, training_decoder_output = \
    model.create_model_data(encoded_training_input, encoded_training_output, output_dict_size)

validation_encoder_input, validation_decoder_input, validation_decoder_output = \
    model.create_model_data(encoded_validation_input, encoded_validation_output, output_dict_size)

"""  delete folder if it exists, and (re)make it 
     also make checkpoints folder  """
version_dir = pathlib.Path('katakana', 'trained_models', training_config['version'])
if version_dir.exists():
    _training_config = getconfig.get_model_config(pathlib.Path('trained_models', training_config['version']))
    _training_config['epochs'] = training_config['epochs']
    training_config = _training_config
else:
    os.mkdir(version_dir)
checkpoints_dir = version_dir.joinpath('checkpoints')
if checkpoints_dir.exists():
    """  just use most recent checkpoint, since epoch index doesnt get saved between runs  """
    latest_checkpoint = max(checkpoints_dir.glob("*-*.hdf5"),
                            key=lambda path: (
                            # int(re.match('.*([0-9]+)-.*.hdf5', str(path)).group(1)),
                            os.path.getmtime(path)))
    seq2seq_model, input_encoding, input_decoding, output_encoding, output_decoding, config = model.load(training_config['version'],
                                                                                                         from_checkpoint_path=latest_checkpoint)
else:
    checkpoints_dir.mkdir()
    latest_checkpoint = None
    # Building the model ----------------------
    seq2seq_model = model.create_model(
        input_dict_size=input_dict_size,
        output_dict_size=output_dict_size,
        input_length=training_config['vector_length'],
        output_length=training_config['vector_length'])

    model.save_config(
        config=training_config)

    model.save_encodings(
        input_encoding=input_encoding,
        input_decoding=input_decoding,
        output_encoding=output_encoding,
        output_decoding=output_decoding,
        config=training_config)

"""  stop when ceases to improve  """
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=3,
                                                        restore_best_weights=True
                                                        )
"""  save all checkpoints  """
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoints_dir.joinpath('{epoch:02d}-{val_loss:.2f}.hdf5'),
    save_weights_only=False,
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
    epochs=training_config['epochs'],
    callbacks=[model_checkpoint_callback, early_stopping_callback])

model.save_model(
    model=seq2seq_model,
    config=training_config)