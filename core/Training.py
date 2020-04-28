import os
import pathlib

from core.Neural_Networks import SimpleNet1
import gym

# CEM imports
from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory


class Trainer:
	def __init__(self, gym_environment=gym.make("forex-v0"), neural_network=SimpleNet1().model):
		# Assume user has their environment and model's input and output both compatible
		self.env = gym_environment
		self.model = neural_network
	
	def reinforce_train_cem(self, steps=60000, visualize=False, verbose=1, nb_steps_warmup=10000,
	                        save_path=r"D:\Data\markets\weights", save_weights_name="cem_CADJPY_weights.h5f"):
		memory = EpisodeParameterMemory(limit=200, window_length=1)
		nb_actions = self.env.action_space.n
		
		agent = CEMAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup)
		agent.compile()
		agent.fit(self.env, nb_steps=steps, visualize=visualize, verbose=verbose)
		
		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
		file_path = os.path.join(save_path, save_weights_name)
		agent.save_weights(filepath=file_path, overwrite=True)
