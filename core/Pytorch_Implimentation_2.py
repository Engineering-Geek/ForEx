import pfrl
import torch
import torch.nn
import gym
import numpy as np
import logging
import sys
import gym_forex
import pandas as pd
from typing import List

env = gym.make('forex-v0')
env.__init__(df=pd.read_pickle("../../data/CADJPY.pkl"))
obs = env.reset()


class QFunction(torch.nn.Module):
	def __init__(self, obs_size, n_actions):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(3, 50, kernel_size=3)
		self.pool1 = torch.nn.MaxPool2d(2)
		self.l2 = torch.nn.Linear(273800, 50)
		self.l3 = torch.nn.Linear(50, n_actions)
		self.relu = torch.nn.ReLU()
	
	def forward(self, x: List[torch.Tensor]):
		h = x
		print(np.shape(h.data))
		h = self.conv1(h)
		h = self.pool1(h)
		h = self.pool1(h).reshape(-1)
		h = self.relu(self.l2(h))
		print(h.shape)
		h = self.l3(h)
		h = torch.Tensor(h.cpu())
		print(np.shape(h))
		return pfrl.action_value.DiscreteActionValue(h)


obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

q_func2 = torch.nn.Sequential(
	torch.nn.Linear(obs_size, 50),
	torch.nn.ReLU(),
	torch.nn.Linear(50, 50),
	torch.nn.ReLU(),
	torch.nn.Linear(50, n_actions),
	pfrl.q_functions.DiscreteActionValueHead(),
)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
gamma = 0.9
phi = lambda x: x[0].astype(np.float32, copy=False)
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
gpu = 0
agent = pfrl.agents.DoubleDQN(
	q_func,
	optimizer,
	replay_buffer,
	gamma,
	explorer,
	phi=phi,
	replay_start_size=500,
	update_interval=1,
	target_update_interval=100,
	gpu=gpu,
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

pfrl.experiments.train_agent_with_evaluation(
	agent,
	env,
	steps=2000,
	eval_n_steps=None,
	eval_n_episodes=10,
	train_max_episode_len=200,
	eval_interval=1000,
	outdir='result'
)
