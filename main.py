from core.Training import Trainer
from core.Model_Environment import ModelAndEnvironment
from model_envs import ModelEnv1

from termcolor import colored
import pandas as pd

filepath = r"D:\Data\markets\MT5\CADJPY.pkl"
# TODO: Create batches based on the maximum window size to train on; integrate with both CEM and DQN.
# --------------------------------------------------------[MAIN]--------------------------------------------------------
# Creates a class dedicated to integrating the model with the feedback from the environment
model_env = ModelAndEnvironment(
	input_shape=(300, 300),
	windows=[16],
	df=pd.read_pickle(filepath),
	fee=0.0
)
print(len(model_env.env.df))
# Given the model and it's subsidiary functions and properties, declare the trainer, and train the model
Trainer(model_env=model_env).reinforce_train_cem(
	nb_steps_warmup=5000,
	steps=10000000,
	log_interval=20000,
	verbose=1
)
