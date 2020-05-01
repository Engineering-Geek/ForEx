from core.Training import Trainer
from core.Neural_Networks import NeuralNet3

import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
	if 'forex' in env:
		print("Remove {} from registry".format(env))
		del gym.envs.registration.registry.env_specs[env]
import gym_forex

from termcolor import colored
import pandas as pd

# select which parent model to use
model_parent = NeuralNet3

# ------------------------------------------------[MAIN]------------------------------------------------
# Create learning environment and the neural network
env = gym.make("forex-v0", input_shape=(300, 300), windows=[32, 256],
               df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl"))
model = model_parent(observation_space=env.observation_space, action_space=env.action_space).model

# Declare and use the trainer to train the model
trainer = Trainer(gym_environment=env, neural_network=model)
trainer.reinforce_train_cem(nb_steps_warmup=20000, steps=1000000, log_interval=10000, verbose=1)

print(colored("[INFO]: PROGRAM FINISHED SUCCESSFULLY", "magenta"))
