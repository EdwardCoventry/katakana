from __future__ import print_function

import pathlib
import warnings

import katakana.model.modeldata

warnings.simplefilter(action='ignore', category=FutureWarning)

import os.path

from katakana import getconfig, model, loadcsvdata, encoding
import keras.callbacks
import re

training_config = getconfig.get_training_config()

# Load and shuffle  ----------------------

data = loadcsvdata.load_csvs(training_config)
data = data.sample(frac=1, random_state=0)

data_input = data['english']
data_output = data['katakana']

data_size = len(data)

train_split_index = int(data_size * 90 / 100)

training_input = data_input[:train_split_index]
training_output = data_output[:train_split_index]
validation_input = data_input[train_split_index:]
validation_output = data_output[train_split_index:]


def get_encoding_length(encoding):
    """  probably the extra 1 is for blank chars  """
    return len(encoding) + 2


# Building the model ----------------------

# delete folder if it exists, and (re)make it
#      also make checkpoints folder
version_dir = pathlib.Path('katakana', 'trained_models', str(training_config['version'])).resolve()
if version_dir.exists() and not list(version_dir.glob('*')):
    _training_config = getconfig.get_model_config(pathlib.Path('trained_models', str(training_config['version'])))
    _training_config['epochs'] = training_config['epochs']
    training_config = _training_config
else:
    # if its doesnt exist or is empty
    version_dir.mkdir(exist_ok=True)
checkpoints_dir = version_dir.joinpath('checkpoints')
if checkpoints_dir.exists() and any(checkpoints_dir.glob(f"*-*.{training_config['file_type']}")):
    """  just use most recent checkpoint, since epoch index doesnt get saved between runs  """
    latest_checkpoint = max(checkpoints_dir.glob(f"*-*.{training_config['file_type']}"),
                            key=lambda path: (
                                # int(re.match('.*([0-9]+)-.*.hdf5', str(path)).group(1)),
                                os.path.getmtime(path)))
    to_katakana_model, input_encoding, input_decoding, output_encoding, output_decoding, config = model.load(
        training_config['version'], from_path=latest_checkpoint)

    input_encoding_length = get_encoding_length(input_encoding)
    output_encoding_length = get_encoding_length(output_encoding)

    # Extract the epoch number from the filename using regex
    match = re.search(r"(\d+)-", str(latest_checkpoint.name))
    assert match
    initial_epoch = int(match.group(1))

else:
    checkpoints_dir.mkdir(exist_ok=True)
    latest_checkpoint = None
    initial_epoch = 0

    # Encoding the dataset ----------------------

    input_encoding, input_decoding = encoding.build_characters_encoding(data_input, training_config)
    output_encoding, output_decoding = encoding.build_characters_encoding(data_output, training_config)

    input_encoding_length = get_encoding_length(input_encoding)
    output_encoding_length = get_encoding_length(output_encoding)

    # Building the model ----------------------
    to_katakana_model = model.create_model(
        input_dict_size=input_encoding_length,
        output_dict_size=output_encoding_length,
        input_length=training_config['vector_length'],
        output_length=training_config['vector_length']
    )

    model.save_config(
        config=training_config)

    model.save_encodings(
        input_encoding=input_encoding,
        input_decoding=input_decoding,
        output_encoding=output_encoding,
        output_decoding=output_decoding,
        config=training_config)

encoded_training_input = encoding.transform(input_encoding, training_input, training_config)
encoded_training_output = encoding.transform(output_encoding, training_output, training_config)
encoded_validation_input = encoding.transform(input_encoding, validation_input, training_config)
encoded_validation_output = encoding.transform(output_encoding, validation_output, training_config)

training_encoder_input, training_decoder_input, training_decoder_output = \
    katakana.model.modeldata.create_model_data(encoded_training_input, encoded_training_output, output_encoding_length)

validation_encoder_input, validation_decoder_input, validation_decoder_output = \
    katakana.model.modeldata.create_model_data(encoded_validation_input, encoded_validation_output,
                                               output_encoding_length)

# stop when ceases to improve (supposedly)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=3,
                                                        restore_best_weights=True
                                                        )
# save all checkpoints
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoints_dir.joinpath('{epoch:02d}-{val_loss:.2f}' + f".{training_config['file_type']}")),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False,
    verbose=1)

# Train the model starting from the correct epoch
to_katakana_model.fit(
    x=[training_encoder_input, training_decoder_input],  # Pass both encoder and decoder inputs
    y=training_decoder_output,  # Target output
    validation_data=([validation_encoder_input, validation_decoder_input], validation_decoder_output),  # Validation data
    verbose=1,
    batch_size=64,
    epochs=training_config['epochs'],
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    initial_epoch=initial_epoch  # Start from the correct epoch number
)

model.save_model(
    model=to_katakana_model,
    config=training_config)
