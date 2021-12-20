import gym
import numpy as np
import pandas as pd
from gym import spaces


class ForexEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self, windows=None, df=pd.DataFrame()):
		if windows is None:
			windows = [256]
		if df.empty:
			df = pd.read_pickle("C:\\Users\\melgi\\Markets\\data\\GBPUSD.pkl")
		
		# Dataframe Preprocessing
		
		self.reward_multiplier = 1
		self.windows = windows
		self.index = max(self.windows) + 1
		self.z_score_desired = 2
		self.correct_reward = 0.95
		# Convert dataframe into numpy arrays
		self.data = df.to_numpy()
		
		sample_observation, _ = self._get_observation()
		observations = {}
		if len(self.windows) > 1:
			for index, observation in enumerate(sample_observation):
				observations.update({
					"Input_{}".format(index): spaces.Box(low=0, high=1, shape=observation.shape)
				})
			self.observation_space = spaces.Dict(sample_observation)
		else:
			self.observation_space = spaces.Box(low=0, high=1, shape=sample_observation.shape)
		self.action_space = spaces.Box(low=0, high=1, shape=(2,))
	
	def _get_observation(self):
		if len(self.windows) > 1:
			raw_observations = []
			observation = []
			for window in self.windows:
				raw_observation = self.data[self.index - window: self.index]
				window_observation = self.data[self.index - (window + 1): self.index]
				for index in range(len(window_observation)):
					window_observation[index] = np.diff(window_observation[index])
					window_observation[index] -= np.min(window_observation)
					window_observation[index] /= np.max(window_observation)
				observation.append(window_observation)
				raw_observations.append(raw_observation)
			observation = np.array(observation)
		else:
			raw_observations = self.data[self.index - self.windows[0]: self.index]
			observation = np.diff(self.data[self.index - (self.windows[0] + 1): self.index], axis=0)
		return observation, raw_observations
	
	def _trade_reward(self, exchange_rate=1.0, action=None):
		reward = None
		if action == 0:
			reward = self.broker.buy(exchange_rate=exchange_rate)
		elif action == 1:
			reward = self.broker.sell(exchange_rate=exchange_rate)
		reward *= self.reward_multiplier
		return reward
	
	def _is_done(self):
		done = True if self.index >= len(self.data) - 2 else False
		return done
	
	def step(self, prediction):
		mean, std_dev = prediction
		if mean - self.z_score_desired * std_dev < self.data[self.index + 1] < mean + self.z_score_desired * std_dev:
			reward = self.correct_reward
		else:
			reward = 1 - self.correct_reward
		info = {
			"reward": reward
		}
		next_observation, next_raw_observation = self._get_observation()
		self.index += 1
		return next_observation, reward, self._is_done(), info
	
	def reset(self):
		self.index = max(self.windows) + 1
		observation, reward, done, info = self.step(1)
		return observation
	
	def render(self, mode='human', close=False):
		pass

