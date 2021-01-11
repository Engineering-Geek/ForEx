from core.PyTorch.RL_Classes import ReplayBuffer, Experience
import gym
import gym_forex
from torch import nn
import numpy as np
import torch
from typing import Tuple


class Agent:
	def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
		self.env = env
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
