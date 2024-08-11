import re
import pandas as pd
from colorama import Fore, Style

def verify_data(data, file_name, config):
    incorrect_pairs = []
    long_text_pairs = []

    def contains_kanji_or_hiragana(text):
        return bool(re.search(r'[\u3040-\u309F\u4E00-\u9FFF]', text))

    def contains_katakana(text):
        return bool(re.search(r'[\u30A0-\u30FF]', text))

    def contains_english(text):
        return bool(re.search(r'[a-zA-Z]', text))

    def contains_digits(text):
        return bool(re.search(r'\d', text))

    def data_is_too_long(text):
        max_length = config['vector_length']
        return len(text) > max_length

    rows_to_remove = data.apply(
        lambda row: (
            contains_digits(row['english']) or
            contains_digits(row['katakana']) or
            data_is_too_long(row['english']) or
            data_is_too_long(row['katakana'])
        ), axis=1
    )

    for idx, row in data[rows_to_remove].iterrows():
        english, katakana = row['english'], row['katakana']

        if contains_digits(english) or contains_digits(katakana):
            print(f"{Fore.YELLOW}  - Row {idx} with digits: English: {english}, Katakana: {katakana}{Style.RESET_ALL}")

        if data_is_too_long(english) or data_is_too_long(katakana):
            long_text_pairs.append((english, katakana))

    data = data[~rows_to_remove].reset_index(drop=True)

    if rows_to_remove.any():
        print(f"{Fore.YELLOW}  Rows removed from file '{file_name}': {rows_to_remove.sum()}{Style.RESET_ALL}")
        if long_text_pairs:
            print(f"{Fore.YELLOW}  Pairs of text that were too long: {long_text_pairs}{Style.RESET_ALL}")

    for idx, row in data.iterrows():
        english, katakana = row['english'], row['katakana']

        if contains_kanji_or_hiragana(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains Kanji or Hiragana")

        if contains_katakana(english) and not contains_english(english):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> English contains Katakana")

        if contains_english(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains English")

    if incorrect_pairs:
        print(f"Data validation errors in file '{file_name}':")
        for pair in incorrect_pairs:
            print(f"  - {pair}")
        assert not incorrect_pairs, f"Data validation errors found: {incorrect_pairs}"