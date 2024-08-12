import re
from colorama import Fore, Style

from katakana.encoding import LEN_START_AND_END_CODES, is_valid, LANGUAGE


def clean_data(data, file_name, config):
    incorrect_pairs = []
    long_text_pairs = []
    rows_to_remove = []

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
        return len(text) + LEN_START_AND_END_CODES > max_length

    def contains_invalid_english(text):
        return not is_valid(text, LANGUAGE.ENGLISH)

    def contains_invalid_katakana(text):
        return not is_valid(text, LANGUAGE.KATAKANA)

    # Get initial length
    initial_length = len(data)

    # Iterate over the rows and identify those to be removed
    for idx, row in data.iterrows():
        english, katakana = row['english'], row['katakana']

        if contains_digits(english) or contains_digits(katakana) or data_is_too_long(english) or data_is_too_long(
                katakana) or contains_invalid_english(english) or contains_invalid_katakana(katakana):
            rows_to_remove.append(idx)
            # Truncate the output if it's too long
            truncated_english = (english[:20] + '...') if len(english) > 23 else english
            truncated_katakana = (katakana[:20] + '...') if len(katakana) > 23 else katakana
            long_text_pairs.append(
                f"{Fore.LIGHTWHITE_EX}{truncated_english}{Fore.LIGHTBLACK_EX},{Fore.LIGHTWHITE_EX}{truncated_katakana}{Style.RESET_ALL}")

    # Print the total number of rows identified for removal
    print(f"{Fore.LIGHTBLACK_EX}Total rows identified for removal: {Fore.CYAN}{len(rows_to_remove)}{Style.RESET_ALL}")

    # Remove the rows that match the condition using data.drop()
    data = data.drop(index=rows_to_remove).reset_index(drop=True)

    # Get final length after cleaning
    final_length = len(data)

    # Print the number of rows removed and the lengths before and after
    rows_removed = initial_length - final_length
    print(
        f"{Fore.LIGHTBLACK_EX}Rows removed from file '{Fore.CYAN}{file_name}{Style.RESET_ALL}': {Fore.RED}{rows_removed}{Style.RESET_ALL} (Initial: {Fore.CYAN}{initial_length}{Style.RESET_ALL}, Final: {Fore.CYAN}{final_length}{Style.RESET_ALL})")

    if rows_removed > 0 and long_text_pairs:
        print(f"{Fore.LIGHTBLACK_EX}Pairs of text that were too long or contained invalid characters:{Style.RESET_ALL}")

        # Truncate each pair of long text in the list
        truncated_long_text_pairs = [
            f"{Fore.LIGHTWHITE_EX}{pair[:20] + '...' if len(pair) > 23 else pair}{Style.RESET_ALL}"
            for pair in long_text_pairs
        ]

        # Join all truncated pairs into a single line
        long_text_pairs_line = f"{'  '.join(truncated_long_text_pairs)}"
        print(f"{long_text_pairs_line}")

    # Validate remaining data for incorrect characters
    for idx, row in data.iterrows():
        english, katakana = row['english'], row['katakana']

        if contains_kanji_or_hiragana(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains Kanji or Hiragana")

        if contains_katakana(english) and not contains_english(english):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> English contains Katakana")

        if contains_english(katakana) and not contains_katakana(katakana):
            incorrect_pairs.append(f"English: {english}, Katakana: {katakana} -> Katakana contains English")

    if incorrect_pairs:
        print(f"{Fore.RED}Data validation errors in file '{file_name}':{Style.RESET_ALL}")
        for pair in incorrect_pairs:
            print(f"  - {Fore.LIGHTRED_EX}{pair}{Style.RESET_ALL}")
        assert not incorrect_pairs, f"{Fore.RED}Data validation errors found: {incorrect_pairs}{Style.RESET_ALL}"

    return data  # Return the cleaned data