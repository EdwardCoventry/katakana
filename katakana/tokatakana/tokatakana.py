from .loaddefaultmodel import load_default_model
from .wordsplit import word_split
from katakana.encoding import format_text, is_valid
from katakana.model import convert_to_katakana

def to_katakana(text, version=None, checkpoint=None, use_tflite=True):

    (loaded_model, input_encoding, input_decoding, output_encoding, output_decoding, model_config, use_model_config) = load_default_model(
        version=version,
        checkpoint=checkpoint,
        use_tflite=use_tflite
    )

    words = word_split(text, model_config)

    # Process each word separately
    converted_words = []
    for word in words:
        if is_valid(word):  # If the word is a digit or symbol, add it directly
            formatted_word = format_text(word, model_config['convert_to_lower'],
                                         model_config['convert_to_unidecode'])
            converted_word = convert_to_katakana(
                text=formatted_word,
                model=loaded_model,
                input_encoding=input_encoding,
                output_decoding=output_decoding,
                config=model_config
            )
            converted_words.append(converted_word)
        else:
            converted_words.append(word)

    # Join the converted words back together with no separator (preserving original format)
    joined_text = ''.join(converted_words)

    return joined_text