import argparse
from abc import ABC
from collections import OrderedDict, deque
from collections import namedtuple
from typing import Tuple

import gym
import gym_forex
import numpy as np
from numpy import array
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from typing import List


class DQN(nn.Module):
	def __init__(self, obs_size: List[int], n_actions: int, hidden_size: int = 128):
		super(DQN, self).__init__()
		self.input = nn.Sequential(
			nn.Linear(obs_size[0], hidden_size),
			nn.ReLU(),
		)
		self.final_layer = nn.Sequential(
			nn.Linear(hidden_size * obs_size[1], n_actions),
		)
	
	def forward(self, batch: Tensor):
		x = batch.float()
		middle_layer = []
		for index in range(len(x[0, 0, :])):
			middle_layer_index = self.input(x[:, :, index])
			middle_layer.append(middle_layer_index)
		middle_layer = torch.cat(middle_layer, 1)
		output = self.final_layer(middle_layer)
		return output


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self.buffer = deque(maxlen=capacity)
	
	def __len__(self) -> int:
		return len(self.buffer)
	
	def append(self, experience: Experience) -> None:
		self.buffer.append(experience)
	
	def sample(self, batch_size: int) -> Tuple:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
		return array(states), array(actions), array(rewards, dtype=np.float32), array(done, dtype=np.bool), array(next_states)


class RLDataset(IterableDataset, ABC):
	def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
		self.buffer = buffer
		self.sample_size = sample_size
	
	def __iter__(self) -> Tuple:
		states, actions, rewards, done, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(done)):
			yield states[i], actions[i], rewards[i], done[i], new_states[i]


class Agent:
	def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
		self.env = env
		self.env.__init__()
		self.replay_buffer = replay_buffer
		self.reset()
		self.state = self.env.reset()
	
	def reset(self) -> None:
		self.state = self.env.reset()
	
	def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state = torch.tensor([self.state])
			
			if device not in ['cpu']:
				state = state.cuda(torch.device(device))
			
			q_values = net(state)
			_, action = torch.max(q_values, dim=1)
			action = int(action.item())
		
		return action
	
	@torch.no_grad()
	def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
		action = self.get_action(net, epsilon, device)
		new_state, reward, done, _ = self.env.step(action)
		exp = Experience(self.state, action, reward, done, new_state)
		self.replay_buffer.append(exp)
		self.state = new_state
		self.reset() if done else None
		return reward, done


class DQNLightning(pl.LightningModule):
	def __init__(self, hparams: argparse.Namespace) -> None:
		super().__init__()
		self.hparams = hparams
		
		self.env = gym.make(self.hparams.env)
		
		self.net = DQN(self.env.observation_space.shape, self.env.action_space.n)
		self.target_net = DQN(self.env.observation_space.shape, self.env.action_space.n)
		
		self.buffer = ReplayBuffer(self.hparams.replay_size)
		self.agent = Agent(self.env, self.buffer)
		self.total_reward = 0
		self.episode_reward = 0
		self.populate(self.hparams.warm_start_steps)
	
	def populate(self, steps: int = 1000) -> None:
		for i in range(steps):
			self.agent.play_step(self.net, epsilon=1.0)
	
	def forward(self, x: Tensor) -> Tensor:
		output = self.net(x)
		return output
	
	def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
		states, actions, rewards, done, next_states = batch
		actions = actions.type(torch.int64)
		state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
		
		with torch.no_grad():
			next_state_values = self.target_net(next_states).max(1)[0]
			next_state_values[done] = 0.0
			next_state_values = next_state_values.detach()
		
		expected_state_action_values = next_state_values * self.hparams.gamma + rewards
		return nn.MSELoss()(state_action_values, expected_state_action_values)
	
	def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], nb_batch) -> OrderedDict:
		device = self.get_device(batch)
		epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame)
		
		reward, done = self.agent.play_step(self.net, epsilon, device)
		self.episode_reward += reward
		loss = self.dqn_mse_loss(batch)
		
		if self.trainer.use_dp or self.trainer.use_ddp2:
			loss = loss.unsqueeze(0)
		
		if done:
			self.total_reward = self.episode_reward
			self.episode_reward = 0
		
		# Soft update of target network
		if self.global_step % self.hparams.sync_rate == 0:
			self.target_net.load_state_dict(self.net.state_dict())
		
		self.log("Total Reward", torch.tensor(self.total_reward).to(device))
		self.log("Steps", torch.tensor(self.global_step).to(device), prog_bar=True)
		self.log("Train Loss", loss, prog_bar=True)
		self.log("Reward", torch.tensor(reward).to(device), prog_bar=True)
		self.log("Episode Reward", torch.tensor(self.episode_reward).to(device), prog_bar=True)
		
		return OrderedDict({"loss": loss})
	
	def configure_optimizers(self) -> List[Optimizer]:
		optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
		return [optimizer]
	
	def __dataloader(self) -> DataLoader:
		dataset = RLDataset(self.buffer, self.hparams.episode_length)
		dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
		return dataloader
	
	def train_dataloader(self) -> DataLoader:
		return self.__dataloader()
	
	def get_device(self, batch) -> str:
		return batch[0].device.index if self.on_gpu else 'cpu'
