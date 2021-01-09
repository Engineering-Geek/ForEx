from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mplfinance.original_flavor import candlestick_ohlc


def get_quotes(df=pd.DataFrame()):
	quotes = []
	index = 1
	for _, row in df.iterrows():
		quotes.append((index, row["Open"], row["Close"], row["High"], row["Low"]))
		index += 1
	quotes = np.asarray(quotes)
	return quotes


def df_candlestick_img(df=pd.DataFrame(), image_dimensions=()):
	quotes = get_quotes(df).astype(float)
	fig = plt.figure(num=0, figsize=(image_dimensions[1] / 80, image_dimensions[0] / 80), dpi=80)
	ax = fig.gca()
	canvas = FigureCanvas(fig)
	candlestick_ohlc(ax, quotes)
	plt.axis("off")
	w, h = fig.canvas.get_width_height()
	canvas.draw()
	img_np_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
	img_np_array.shape = (w, h, 3)
	img_np_array = np.reshape(np.array(Image.fromarray(img_np_array).convert("L")),
	                          (image_dimensions[0], image_dimensions[1])) / 255.0
	img_np_array = np.asarray([img_np_array, img_np_array, img_np_array])
	img_np_array = np.reshape(img_np_array, (image_dimensions[0], image_dimensions[1], 3))
	
	fig.clf()
	return img_np_array / 255.0


def df_line_img(df=pd.DataFrame(), image_dimensions=()):
	fig = plt.figure(num=0, figsize=(image_dimensions[1] / 80, image_dimensions[0] / 80), dpi=80)
	canvas = FigureCanvas(fig)
	df["Open"].plot()
	plt.axis("off")
	w, h = fig.canvas.get_width_height()
	canvas.draw()
	img_np_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
	img_np_array.shape = (w, h, 3)
	
	fig.clf()
	return np.asarray(img_np_array / 255.0)


class BankAccount:
	def __init__(self, starting=2000.00):
		self.balance = starting
		self.currency1 = starting
		self.currency2 = 0
		self.last_exchange_rate = 1.0
	
	def calculate_balance(self, exchange_rate=1.0):
		if self.currency1 == 0:
			self.balance = self.currency2 / exchange_rate
		else:
			self.balance = self.currency1
		self.last_exchange_rate = exchange_rate
		return self.balance
	
	def transfer_1_2(self, exchange_rate=1.0):
		self.currency2 = self.currency1 * exchange_rate
		self.currency1 = 0
		self.last_exchange_rate = exchange_rate
	
	def transfer_2_1(self, exchange_rate=1.0):
		self.currency1 = self.currency2 / exchange_rate
		self.currency2 = 0
		self.last_exchange_rate = exchange_rate


class Broker:
	def __init__(self, transaction_fee=0.01, account=BankAccount()):
		# 1% trading fee per trade
		self.account = account
		self.transaction_fee = transaction_fee
		self.previous_balance = self.account.currency1
		self.order_number = 0
	
	def buy(self, exchange_rate):
		if self.account.currency2 == 0:
			self.order_number += 1
			# Take the transaction fee
			self.account.currency1 *= (1 - self.transaction_fee)
			# Do transaction
			self.account.transfer_1_2(exchange_rate)
		# calculate reward
		current_balance = self.account.calculate_balance(exchange_rate)
		dif = current_balance - self.previous_balance
		reward = 2 * dif / (current_balance + self.previous_balance)
		self.previous_balance = current_balance
		return reward
	
	def sell(self, exchange_rate):
		if self.account.currency1 == 0:
			self.order_number += 1
			# Take the transaction fee
			self.account.currency2 *= (1 - self.transaction_fee)
			# Do transaction
			self.account.transfer_2_1(exchange_rate)
		# calculate reward
		current_balance = self.account.calculate_balance(exchange_rate)
		dif = current_balance - self.previous_balance
		reward = 2 * dif / (current_balance + self.previous_balance)
		self.previous_balance = current_balance
		return reward
