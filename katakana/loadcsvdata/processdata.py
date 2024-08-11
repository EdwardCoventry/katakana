from katakana.encoding.formattext import format_text
from katakana.loadcsvdata.verifydata import verify_data
from katakana.loadcsvdata.loadfiles import load_csv_files

def load_csvs(config):
    all_data = load_csv_files()

    print(f"  Initial number of rows: {len(all_data)}")

    convert_to_lower = config['convert_to_lower']
    convert_to_unidecode = config['convert_to_unidecode']

    all_data = all_data.dropna()

    """ Format data according to config """
    for column in ['english', 'katakana']:
        all_data[column] = all_data[column].apply(lambda x: format_text(str(x), convert_to_lower, convert_to_unidecode))

    print(f"  After formatting: {len(all_data)} rows")

    # Verify data integrity and remove rows with digits
    for file_name in all_data['file_name'].unique():
        print(f"Processing file: {file_name}")
        data = all_data[all_data['file_name'] == file_name]
        verify_data(data, file_name, config)
        all_data.update(data)

    print(f"  After verification: {len(all_data)} rows")

    return all_data