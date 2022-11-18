import glob
import pandas as pd


def load_csvs():
	"""  will read all csvs from data folder  """
	files = glob.glob("./dataset/data/*.csv")
	dfs = [pd.read_csv(f, header=None, sep=",") for f in files]
	data = pd.concat(dfs, ignore_index=True, verify_integrity=True)
	data.columns = ['english', 'katakana']
	data = data.dropna()
	data = data.drop_duplicates()
	data = data.reset_index(drop=True)
	return data
