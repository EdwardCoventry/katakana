import glob
import pandas as pd

def load_csv_files():
    """ Will read all CSVs from the data folder """
    files = glob.glob("./dataset/data/*.csv")

    all_data = pd.DataFrame()

    for file in files:
        print(f"Loading file: {file}")
        data = pd.read_csv(file, header=None, sep=",")
        data.columns = ['english', 'katakana']
        data['file_name'] = file  # Add file_name column to keep track of the source file
        all_data = pd.concat([all_data, data], ignore_index=True, verify_integrity=True)

    return all_data