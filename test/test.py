import pandas as pd

pd.read_pickle(r"D:\Data\markets\CADJPY_2.pkl")[:10000].to_pickle(r"D:\Data\markets\CADJPY_3.pkl")
