import glob
import pandas as pd
from katakana import formattext

def load_csvs(config):
	"""  will read all csvs from data folder  """
	files = glob.glob("./dataset/data/*.csv")
	dfs = [pd.read_csv(f, header=None, sep=",") for f in files]
	data = pd.concat(dfs, ignore_index=True, verify_integrity=True)
	data.columns = ['english', 'katakana']

	convert_to_lower = config['convert_to_lower']
	convert_to_unidecode = config['convert_to_unidecode']

	data = data.dropna()

	"""  format data according to config """
	for i, column in enumerate(['english', 'katakana']):
		data[column] = [
			formattext.format_text(str(x), convert_to_lower, convert_to_unidecode)
			for x in data[column].to_list()
		]

	data = data.dropna()
	data = data.drop_duplicates()
	# delete any rows with any digits
	any_digits = data[(data['english'].str.contains(r'\d')) | (data['katakana'].str.contains(r'\d'))].index
	data = data.drop(any_digits)
	data = data.reset_index(drop=True)
	return data
