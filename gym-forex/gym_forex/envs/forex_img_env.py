import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym_forex.envs.forex_functions import Broker, BankAccount, df_line_img, df_candlestick_img


class ForexEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self, input_shape=(299, 299, 3), nb_actions=2, windows=None, df=pd.DataFrame(), fee=0.00):
		if windows is None:
			windows = [64]
		self.image_shape = [input_shape[0], input_shape[1]]
		self.fee = fee
		
		self.broker = Broker(transaction_fee=self.fee, account=BankAccount())
		self.windows = windows
		self.index = max(self.windows) + 1
		self.df = df
		self.custom_df = None
		
		image_spaces = self._get_observation()
		observations = {}
		index = 0
		for observation in image_spaces:
			observations.update({
				"Input_{}".format(index): spaces.Box(low=0, high=1, shape=np.shape(observation))
			})
			index += 1
		if len(observations.values()) > 1:
			self.observation_space = spaces.Dict(image_spaces)
		else:
			self.observation_space = list(observations.values())[0]
		self.action_space = spaces.Discrete(nb_actions)
	
	def _get_observation(self):
		images = []
		first = True
		for window in self.windows:
			if first:
				img = df_candlestick_img(df=self.df.iloc[self.index - window: self.index],
				                         image_dimensions=self.image_shape)
				first = False
			else:
				img = df_line_img(df=self.df.iloc[self.index - window: self.index], image_dimensions=self.image_shape)
			images.append(img)
		return images
	
	def _trade_reward(self, exchange_rate=1.0, action=None):
		reward = None
		if action is None:
			TypeError("'action' has no value passed")
		if action == 0:
			reward = self.broker.buy(exchange_rate=exchange_rate)
		elif action == 1:
			reward = self.broker.sell(exchange_rate=exchange_rate)
		return reward
	
	def _is_done(self):
		done = False
		if self.index >= len(self.df) - 1:
			done = True
		return done
	
	def step(self, action):
		current_row = self.df.iloc[self.index]
		# Getting DataFrame window and it's images
		observation = self._get_observation()
		# Trade and get respective reward
		reward = self._trade_reward(current_row["Close"], action)
		# Is done?
		done = self._is_done()
		# Summarize all this as info
		info = {
			"reward": reward,
			"number of trades": self.broker.order_number,
			"account balance": self.broker.account.calculate_balance(exchange_rate=current_row["Close"])
		}
		# Increase index so we can calculate the next observation set
		self.index += 1
		return observation, reward, done, info
	
	def reset(self):
		self.broker = Broker(transaction_fee=self.fee, account=BankAccount())
		self.index = max(self.windows) + 1
		images, reward, done, info = self.step(1)
		return images
	
	def render(self, mode='human', close=False):
		pass
