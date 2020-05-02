import os
import pathlib
import shutil

from core.Neural_Networks import SimpleNet1
import gym

# CEM imports
from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.memory import EpisodeParameterMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


class Trainer:
	def __init__(self, gym_environment=gym.make("forex-v0"), neural_network=SimpleNet1().model):
		# Assume user has their environment and model's input and output both compatible
		self.env = gym_environment
		self.model = neural_network

	def _create_empty_directory(self, filepath):
		dirpath = pathlib.Path(filepath)
		if dirpath.exists() and dirpath.is_dir():
			shutil.rmtree(dirpath)
		dirpath.mkdir(parents=True, exist_ok=True)

	def callbacks(self):
		# Creating file paths
		training_save_folder = r"/media/melgiri/Storage/Data/markets/"

		checkpoint_path = os.path.join(training_save_folder, "checkpoints")
		self._create_empty_directory(checkpoint_path)
		logs_path = os.path.join(training_save_folder, "logs")
		self._create_empty_directory(logs_path)

		callbacks = [
			ModelIntervalCheckpoint(
				filepath=os.path.join(checkpoint_path, "weights.h5"),
				verbose=0,
				interval=64 ** 2
			),
			FileLogger(
				filepath=os.path.join(logs_path, "logs.json"),
				interval=64
			)
		]

		return callbacks

	def reinforce_train_cem(self, steps=60000, visualize=False, verbose=1, nb_steps_warmup=1000,
	                        save_path=r"/media/melgiri/Storage/Data/markets",
	                        save_weights_name="cem_CADJPY_weights.h5f"):
		memory = EpisodeParameterMemory(limit=200, window_length=1)
		nb_actions = self.env.action_space.n

		agent = CEMAgent(
			model=self.model,
			nb_actions=nb_actions,
			memory=memory,
			nb_steps_warmup=nb_steps_warmup,
			batch_size=64
		)
		agent.compile()
		agent.fit(
			self.env,
			nb_steps=steps,
			visualize=visualize,
			verbose=verbose,
			callbacks=self.callbacks()
		)

		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
		file_path = os.path.join(save_path, save_weights_name)
		agent.save_weights(filepath=file_path, overwrite=True)

	def reinforce_train_dqn(self, steps=60000, visualize=False, verbose=1, nb_steps_warmup=10000,
	                        save_path=r"/media/melgiri/Storage/Data/markets",
	                        save_weights_name="cem_CADJPY_weights.h5f"):
		memory = EpisodeParameterMemory(limit=200, window_length=1)
		nb_actions = self.env.action_space.n
		policy = LinearAnnealedPolicy(
			inner_policy=EpsGreedyQPolicy(),
			attr="eps",
			value_max=1.,
			value_min=.1,
			value_test=0.5,
			nb_steps=10000,
		)

		agent = DQNAgent(
			model=self.model,
			nb_actions=nb_actions,
			memory=memory,
			nb_steps_warmup=nb_steps_warmup,
			batch_size=64,
			target_model_update=1e-2,
			policy=policy
		)
		Adam._name = "hey"
		agent.compile(
			optimizer=Adam
		)
		agent.fit(
			self.env,
			nb_steps=steps,
			visualize=visualize,
			verbose=verbose,
			callbacks=self.callbacks()
		)

		pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
		file_path = os.path.join(save_path, save_weights_name)
		agent.save_weights(filepath=file_path, overwrite=True)
