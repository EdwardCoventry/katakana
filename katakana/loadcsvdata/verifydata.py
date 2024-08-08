import re
from colorama import Fore

def verify_data(data, file_name):
    incorrect_pairs = []
    rows_with_digits = []

    def contains_kanji_or_hiragana(text):
        return bool(re.search(r'[\u3040-\u309F\u4E00-\u9FFF]', text))

    def contains_katakana(text):
        return bool(re.search(r'[\u30A0-\u30FF]', text))

    def contains_english(text):
        return bool(re.search(r'[a-zA-Z]', text))

    def contains_digits(text):
        return bool(re.search(r'\d', text))

    for idx, row in data.iterrows():
        english, katakana = row['english'], row['katakana']

        if contains_digits(english) or contains_digits(katakana):
            rows_with_digits.append(idx)
            print(f"{Fore.YELLOW}  - Row {idx} with digits: English: {english}, Katakana: {katakana}{Fore.RESET}")

        if contains_kanji_or_hiragana(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains Kanji or Hiragana")

        if contains_katakana(english) and not contains_english(english):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> English contains Katakana")

        if contains_english(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains English")

    if rows_with_digits:
        print(f"{Fore.YELLOW}  Rows with digits in file '{file_name}': {len(rows_with_digits)}")
        data = data.drop(rows_with_digits).reset_index(drop=True)

    if incorrect_pairs:
        print(f"Data validation errors in file '{file_name}':")
        for pair in incorrect_pairs:
            print(f"  - {pair}")
        assert not incorrect_pairs, f"Data validation errors found: {incorrect_pairs}"

    return data
