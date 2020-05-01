import os
import pathlib

import gym
import numpy as np
# CEM imports
from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from rl.processors import Processor

from core.Neural_Networks import SimpleNet1


class MultiInputProcessor(Processor):
	def __init__(self, nb_inputs):
		self.nb_inputs = nb_inputs
	
	def process_state_batch(self, state_batch):
		input_batches = [[] for _ in range(self.nb_inputs)]
		for state in state_batch:
			processed_state = [[] for _ in range(self.nb_inputs)]
			for observation in state:
				assert len(observation) == self.nb_inputs
				for o, s in zip(observation, processed_state):
					s.append(o)
			for idx, s in enumerate(processed_state):
				input_batches[idx].append(s)
		return [np.array(x)[0] for x in input_batches]


class Trainer:
	def __init__(self, gym_environment=gym.make("forex-v0"), neural_network=SimpleNet1().model):
		# Assume user has their environment and model's input and output both compatible
		self.env = gym_environment
		self.model = neural_network
	
	def reinforce_train_cem(self, steps=60000, visualize=False, verbose=1, nb_steps_warmup=10000,
	                        save_path=r"D:\Data\markets\weights", save_weights_name="cem_CADJPY_weights.h5f",
	                        log_interval=1000):
		memory = EpisodeParameterMemory(limit=200, window_length=1)
		nb_actions = self.env.action_space.n
		
		agent = CEMAgent(
			model=self.model,
			nb_actions=nb_actions,
			memory=memory,
			nb_steps_warmup=nb_steps_warmup,
			processor=MultiInputProcessor(nb_inputs=len(self.model.inputs)))
		agent.compile()
		agent.fit(self.env, nb_steps=steps, visualize=visualize, verbose=verbose, log_interval=log_interval)
		
		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
		file_path = os.path.join(save_path, save_weights_name)
		agent.save_weights(filepath=file_path, overwrite=True)
