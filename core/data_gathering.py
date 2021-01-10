import MetaTrader5 as MT5
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import os


def format_df(df=pd.DataFrame()):
	pass


TICKS = 1 * (10 ** 4)  # 1 million ticks; 1 second ticks
base_directory = r"C:\Users\melgi\Markets\data"
Path(base_directory).mkdir(parents=True, exist_ok=True)

if not MT5.initialize():
	print("initialize() failed")
	MT5.shutdown()

CURRENCY_PAIRS = ["CADJPY", "EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
for currency_pair in CURRENCY_PAIRS:
	ticks = MT5.copy_rates_from_pos(currency_pair, MT5.TIMEFRAME_M5, 0, 99999)
	df = pd.DataFrame(ticks)
	df = df.drop(columns=["real_volume", "spread", "tick_volume"])
	df["time"] = pd.to_datetime(df["time"], unit="s")
	df = df.set_index("time")
	df.columns = ["Open", "High", "Low", "Close"]
	# save df
	df.to_pickle(os.path.join(base_directory, "{}.pkl".format(currency_pair)))

print("FINISHED")
