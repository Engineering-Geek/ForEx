from core.Training import Trainer
from model_envs import *

from termcolor import colored
import pandas as pd

filepath = r"D:\Data\markets\MT5\CADJPY.pkl"

# --------------------------------------------------------[MAIN]--------------------------------------------------------
# Given the learning environment, a model will be created with some other functions
model_env = ModelEnv1(input_shape=(300, 300), windows=[32, 16], df=pd.read_pickle(filepath))
# Given the model and it's subsidiary functions and properties, declare the trainer, and train the model
Trainer(model_env=model_env).reinforce_train_cem(nb_steps_warmup=20000, steps=1000000, log_interval=5000, verbose=1)
