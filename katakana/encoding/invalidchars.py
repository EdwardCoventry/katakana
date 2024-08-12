import re
class LANGUAGE:
    ENGLISH = 'english'
    KATAKANA = 'katakana'

# Patterns for English
valid_chars_english = r"a-zA-Z'"
valid_pattern_english = re.compile(f"[{valid_chars_english}]+")
invalid_pattern_english = re.compile(f"[^{valid_chars_english}]+")

# Patterns for Katakana
katakana_chars = r"\u30A0-\u30FF"
valid_pattern_katakana = re.compile(f"[{katakana_chars}]")
invalid_pattern_katakana = re.compile(f"[^{katakana_chars}]+")

def is_valid(text, language=LANGUAGE.ENGLISH):
    """
    Checks if the text contains only valid characters according to the specified language.

    :param text: The text to be checked.
    :param language: The language to check against ('english' or 'katakana').
    :return: True if the text is valid, False otherwise.
    """
    if language == LANGUAGE.ENGLISH:
        return not bool(invalid_pattern_english.search(text))
    elif language == LANGUAGE.KATAKANA:
        return not bool(invalid_pattern_katakana.search(text))
    else:
        raise ValueError(f"Unsupported language: {language}")


def split_at_valid(text, language=LANGUAGE.ENGLISH):
    """
    Splits the text into parts based on valid characters according to the specified language.

    :param text: The text to be split.
    :param language: The language to split by ('english' or 'katakana').
    :return: List of valid character parts, excluding empty strings.
    """
    if language == LANGUAGE.ENGLISH:
        parts = valid_pattern_english.findall(text)
    elif language == LANGUAGE.KATAKANA:
        parts = valid_pattern_katakana.findall(text)
    else:
        raise ValueError(f"Unsupported language: {language}")

    return [part for part in parts if part]  # Filter out empty strings