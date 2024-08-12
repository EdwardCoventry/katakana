from katakana.encoding.formattext import format_text
from katakana.loadcsvdata.cleandata import clean_data
from katakana.loadcsvdata.loadfiles import load_csv_files
import pandas as pd
from colorama import Fore, Style

def load_csvs(config):
    all_data = load_csv_files()

    print(f"{Fore.MAGENTA}  Initial number of rows: {len(all_data)}{Style.RESET_ALL}")

    convert_to_lower = config['convert_to_lower']
    convert_to_unidecode = config['convert_to_unidecode']

    all_data = all_data.dropna()

    """ Format data according to config """
    for column in ['english', 'katakana']:
        all_data[column] = all_data[column].apply(lambda x: format_text(str(x), convert_to_lower, convert_to_unidecode))

    length_before_cleaning = len(all_data)

    print(f"{Fore.MAGENTA}  Before cleaning: {length_before_cleaning} rows{Style.RESET_ALL}")

    # Initialize an empty DataFrame to collect cleaned data
    cleaned_data = []

    # Verify data integrity and remove rows with digits
    for file_name in all_data['file_name'].unique():
        print(f"Processing file: {file_name}")
        data = all_data[all_data['file_name'] == file_name]
        data = clean_data(data, file_name, config)
        cleaned_data.append(data)

    # Concatenate all cleaned data into a single DataFrame
    all_data = pd.concat(cleaned_data, ignore_index=True)

    length_after_cleaning = len(all_data)

    assert length_after_cleaning > 0, "No data left after cleaning"

    # Calculate total pairs removed
    pairs_removed = length_before_cleaning - length_after_cleaning

    print(f"{Fore.MAGENTA}  After cleaning: {length_after_cleaning} rows ({pairs_removed} pairs removed){Style.RESET_ALL}")

    return all_data
