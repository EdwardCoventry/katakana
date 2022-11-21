from __future__ import print_function

from katakana import getconfig, model, encoding, loadcsvdata

# ===============================================================

# Read YAML file
use_model_config = getconfig.get_use_model_config()

print('Loading the model...')

testing_model, input_encoding, input_decoding, output_encoding, output_decoding, config = model.load(use_model_config['version'])

# ===============================================================

print('Evaluating the model on random testing dataset...')

data = loadcsvdata.load_csvs()
data = data.sample(frac=1, random_state=11)

data_input = data['english']
data_output = data['katakana']

data_size = len(data)
test_split = int(data_size*10/100)

test_input = data_input[:test_split]
test_output = data_output[:test_split]

encoded_testing_input = encoding.transform(input_encoding, test_input, config)
encoded_testing_output = encoding.transform(output_encoding, test_output, config)

test_encoder_input, test_decoder_input, test_decoder_output = \
    model.create_model_data(encoded_testing_input, encoded_testing_output, len(output_decoding) + 1)

testing_model.evaluate(x=[test_encoder_input, test_decoder_input], y=test_decoder_output)

# ===============================================================

print('Evaluating the model on random names...')

config = model.load_config(use_model_config['version'])

def to_katakan(english_text):
    return model.to_katakana(english_text, testing_model, input_encoding, output_decoding, config=config)


print(data_input[0], to_katakan(data_input[0]))
print(data_input[1], to_katakan(data_input[1]))
print(data_input[2], to_katakan(data_input[2]))


print('Hello World', to_katakan('Hello World'), '-', 'カナダ')
print('Banana', to_katakan('Banana'), '-', 'バナナ')
print('Test', to_katakan('Test'), '-', '')
print('Canada', to_katakan('Canada'), '-', '')
print('Barbecue', to_katakan('Barbecue'), '-', 'バーベキュー')
print('Google Maps', to_katakan('Google Maps'), '-', 'グーグル マップ')
print('John Doe', to_katakan('John Doe'), '-', 'ジョン・ドウ')
print('Donald Duck', to_katakan('Donald Duck'), '-', 'ドナルド・ダック')
print('Donald Trump', to_katakan('Donald Trump'), '-', 'ドナルド・トランプ ')