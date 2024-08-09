from __future__ import print_function

import katakana.model.modeldata
from katakana import getconfig, model, loadcsvdata
from katakana.encoding import encoding

# ===============================================================
# Load model and configuration
# ===============================================================

use_model_config = getconfig.get_use_model_config()

print('Loading the model...')
testing_model, input_encoding, input_decoding, output_encoding, output_decoding, config = model.loadmodel.load(use_model_config['version'])

# ===============================================================
# Evaluate the model on the dataset
# ===============================================================

print('Evaluating the model on random testing dataset...')

data = loadcsvdata.load_csvs(config).sample(frac=1, random_state=11)
data_input = data['english']
data_output = data['katakana']

test_split = int(len(data) * 0.1)
test_input = data_input[:test_split]
test_output = data_output[:test_split]

encoded_testing_input = encoding.transform(input_encoding, test_input, config)
encoded_testing_output = encoding.transform(output_encoding, test_output, config)

test_encoder_input, test_decoder_input, test_decoder_output = \
    katakana.model.modeldata.create_model_data(encoded_testing_input, encoded_testing_output, len(output_decoding) + 1)

testing_model.evaluate(x=[test_encoder_input, test_decoder_input], y=test_decoder_output)

# ===============================================================
# Define utility functions
# ===============================================================

print('Evaluating the model on random names...')

config = model.load_config(use_model_config['version'])

def to_katakan(english_text):
    return model.to_katakana(english_text, testing_model, input_encoding, output_decoding, config=config)

def print_katakana_pairs(data, katakana_func):
    for item in data:
        print(item, katakana_func(item))

def print_test_cases(katakana_func):
    test_cases = [
        ('Hello World', 'カナダ'),
        ('Banana', 'バナナ'),
        ('Test', ''),
        ('Canada', ''),
        ('Barbecue', 'バーベキュー'),
        ('Google Maps', 'グーグル マップ'),
        ('John Doe', 'ジョン・ドウ'),
        ('Donald Duck', 'ドナルド・ダック'),
        ('Donald Trump', 'ドナルド・トランプ ')
    ]
    for name, expected in test_cases:
        print(name, katakana_func(name), '-', expected)

# ===============================================================
# Main execution
# ===============================================================

print_katakana_pairs(data_input, to_katakan)
print_test_cases(to_katakan)
