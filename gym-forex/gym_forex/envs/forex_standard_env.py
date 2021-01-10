import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym_forex.envs.forex_functions import Broker, BankAccount


class ForexEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self, nb_actions=2, windows=None, df=pd.DataFrame(), fee=0.00):
		if windows is None:
			windows = [128]
		if df.empty:
			df = pd.read_pickle("C:\\Users\\melgi\\Markets\\data\\CADJPY.pkl")
		
		self.reward_multiplier = 1e5
		self.fee = fee
		self.broker = Broker(transaction_fee=self.fee, account=BankAccount())
		self.windows = windows
		self.index = max(self.windows) + 1
		# Convert dataframe into numpy arrays
		self.data = df.to_numpy()
		
		sample_observation = self._get_observation()
		observations = {}
		if len(self.windows) > 1:
			for index, observation in enumerate(sample_observation):
				observations.update({
					"Input_{}".format(index): spaces.Box(low=0, high=1, shape=observation.shape)
				})
			self.observation_space = spaces.Dict(sample_observation)
		else:
			self.observation_space = spaces.Box(low=0, high=1, shape=sample_observation.shape)
		self.action_space = spaces.Discrete(nb_actions)
	
	def _get_observation(self):
		if len(self.windows) > 1:
			observation = []
			for window in self.windows:
				window_observation = self.data[self.index - window: self.index]
				for index in range(len(window_observation)):
					window_observation[index] -= np.min(window_observation)
					window_observation[index] /= np.max(window_observation)
				observation.append(window_observation)
			observation = np.array(observation)
		else:
			observation = self.data[self.index - self.windows[0]: self.index]
			f = lambda x: ((x - min(observation.reshape(-1))) / max(observation.reshape(-1)))
			observation = f(observation)
		return observation
	
	def _trade_reward(self, exchange_rate=1.0, action=None):
		reward = None
		TypeError("'action' has no value passed") if action is None else ""
		if action == 0:
			reward = self.broker.buy(exchange_rate=exchange_rate)
		elif action == 1:
			reward = self.broker.sell(exchange_rate=exchange_rate)
		reward *= self.reward_multiplier
		return reward
	
	def _is_done(self):
		done = True if self.index >= len(self.data) - 1 else False
		return done
	
	def step(self, action):
		reward = self._trade_reward(self.data[self.index][3], action)
		info = {
			"reward": reward,
			"number of trades": self.broker.order_number,
			"account balance": self.broker.account.calculate_balance(exchange_rate=self.data[self.index][3])
		}
		next_observation = self._get_observation()
		self.index += 1
		return next_observation, reward, self._is_done(), info
	
	def reset(self):
		self.broker = Broker(transaction_fee=self.fee, account=BankAccount())
		self.index = max(self.windows) + 1
		observation, reward, done, info = self.step(1)
		return observation
	
	def render(self, mode='human', close=False):
		pass


if __name__ == '__main__':
	ForexEnv(df=pd.read_pickle("../../../../data/CADJPY.pkl"))
