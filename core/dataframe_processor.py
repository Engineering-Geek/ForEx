import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


class DataProcessor:
	def __init__(self, pickle_path="/media/melgiri/Storage/Data/markets/CADJPY.pkl", freq="Min"):
		self.df = pd.read_pickle(filepath_or_buffer=pickle_path)
		self.df = self.df.set_index("time")
		self.df_final = self._open_high_low_close(
			type_="ask",
			freq=freq
		)
	
	# self.bid_df = self._open_high_low_close(
	# 	type_="bid",
	# 	freq=freq
	# )
	
	def _open_high_low_close(self, type_="ask", freq="Min"):
		series = self.df.groupby(pd.Grouper(freq=freq)).progress_apply(
			lambda x:
			[x[type_][0], np.max(x[type_]), np.min(x[type_]), x[type_][-1]] if len(x) > 1
			else [x[type_], x[type_], x[type_], x[type_]]
		)
		df = series.to_frame()
		df.columns = ["col1"]
		df_final = pd.DataFrame(df.col1.to_list(), columns=["Open", "High", "Low", "Close"])
		df_final.index = df.index
		return df_final
