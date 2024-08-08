import glob
import pandas as pd
import re
from katakana import formattext
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

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
            print(f"{Fore.YELLOW}  - Row {idx} with digits: English: {english}, Katakana: {katakana}")

        if contains_kanji_or_hiragana(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains Kanji or Hiragana")

        if contains_katakana(english) and not contains_english(english):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> English contains Katakana")

        if contains_english(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains English")

    if rows_with_digits:
        print(f"{Fore.YELLOW}  Rows with digits in file '{file_name}': {len(rows_with_digits)}")
        data.drop(rows_with_digits, inplace=True)
        data.reset_index(drop=True, inplace=True)

    if incorrect_pairs:
        print(f"Data validation errors in file '{file_name}':")
        for pair in incorrect_pairs:
            print(f"  - {pair}")
        assert not incorrect_pairs, f"Data validation errors found: {incorrect_pairs}"

def load_csvs(config):
    """ Will read all CSVs from the data folder """
    files = glob.glob("./dataset/data/*.csv")

    all_data = pd.DataFrame()

    for file in files:
        print(f"Processing file: {file}")
        dfs = [pd.read_csv(file, header=None, sep=",")]
        data = pd.concat(dfs, ignore_index=True, verify_integrity=True)
        data.columns = ['english', 'katakana']

        print(f"  Initial number of rows: {len(data)}")

        convert_to_lower = config['convert_to_lower']
        convert_to_unidecode = config['convert_to_unidecode']

        data = data.dropna()

        """ Format data according to config """
        for i, column in enumerate(['english', 'katakana']):
            data[column] = [
                formattext.format_text(str(x), convert_to_lower, convert_to_unidecode)
                for x in data[column].to_list()
            ]

        print(f"  After formatting: {len(data)} rows")

        # Verify data integrity and remove rows with digits
        verify_data(data, file)

        print(f"  After verification: {len(data)} rows")

        all_data = pd.concat([all_data, data], ignore_index=True, verify_integrity=True)

    return all_data
