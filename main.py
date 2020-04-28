# Importing custom classes and functions
from core.dataframe_processor import DataProcessor
from core.Training import Trainer
from core.Neural_Networks import SimpleNet1

from gym_anytrading.envs import ForexEnv
from termcolor import colored

import os

# clear TensorFlow declaration warnings
os.system('cls')

# select which parent model to use
model_parent = SimpleNet1


# Specialized environment for gym as specified by the parent model above
class SpecializedForexEnv(ForexEnv):
	# the specialized_data_funnel function is somewhat static but is inside this class to help organize data
	#   processors with models
	_process_data = model_parent.specialized_data_funnel


# ------------------------------------------------[MAIN]------------------------------------------------
print(colored("[INFO]: Generating Data for environment", "green"))
data = DataProcessor(
	pickle_path=r"D:\Data\markets\CADJPY.pkl"
)

print(colored("[INFO]: Creating ForEx environment for training", "green"))
env = SpecializedForexEnv(
	df=data.df_final,
	window_size=12,
	frame_bound=(12, len(data.df_final))
)

print(colored("[INFO]: Building Neural Network", "green"))
model = model_parent().build_model(
	input_size=env.observation_space.shape,
	nb_actions=env.action_space.n
)

print(colored("[INFO]: Declaring trainer", "green"))
trainer = Trainer(
	gym_environment=env,
	neural_network=model
)

print(colored("[INFO]: Cross entropy reinforcement training", "green"))
trainer.reinforce_train_cem(
	nb_steps_warmup=20000,
	steps=100000
)

print(colored("[INFO]: PROGRAM FINISHED SUCCESSFULLY", "magenta"))
