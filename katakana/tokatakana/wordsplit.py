import wordninja
from katakana.encoding import LEN_START_AND_END_CODES
from katakana.encoding.invalidchars import is_valid, split_at_valid

def word_split(text, config):
    max_length = config['vector_length'] - LEN_START_AND_END_CODES

    # Split the text into words, groups of digits, and groups of non-word characters
    parts = split_at_valid(text)

    result = []

    for part in parts:
        if is_valid(part):
            result.append(part)
        else:
            result.extend(split_section([part], max_length))

    return result

def split_section(text_list, max_length):
    result = []
    for section in text_list:
        if len(section) > max_length:
            # Split the section using wordninja
            split_word = wordninja.split(section)
            # Reconstitute split words based on max_length
            reconstituted = reconstitute_split_word(split_word, max_length)
            # Recursively process each reconstituted part
            result.extend(split_section(reconstituted, max_length))
        else:
            result.append(section)
    return result


def reconstitute_split_word(split_word, max_length):
    result = []
    current_part = ""

    for word in split_word:
        if len(word) > max_length:
            # If a single word exceeds max_length, split it manually with preference for natural breaks
            result.extend(preferred_split(word, max_length))
        elif len(current_part) + len(word) <= max_length:
            current_part += word
        else:
            if current_part:
                result.append(current_part)
            current_part = word

    if current_part:
        result.append(current_part)

    return result


def preferred_split(word, max_length):
    vowels = "aeiou"
    splits = []
    start = 0

    while start < len(word):
        end = min(start + max_length, len(word))

        # Try to find a natural break (vowel before consonant) within the current segment
        split_point = end
        for i in range(end - 1, start, -1):
            if word[i] in vowels and i + 1 < len(word) and word[i + 1] not in vowels:
                split_point = i + 1
                break

        splits.append(word[start:split_point])
        start = split_point

    return splits