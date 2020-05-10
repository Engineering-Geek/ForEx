from core.Training import Trainer
from core.Model_And_Environments import ModelAndEnvironment

from termcolor import colored
import pandas as pd

# ------------------------------------------------[MAIN]------------------------------------------------
# Given the learning environment, a model will be created with some other functions
model_env = ModelAndEnvironment(input_shape=(300, 300), windows=[32],
                                df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl"))
# Given the model and it's subsidiary functions and properties, declare the trainer
trainer = Trainer(model_env=model_env)
# Train the model
trainer.reinforce_train_cem(nb_steps_warmup=20000, steps=1000000, log_interval=5000, verbose=1)

print(colored("[INFO]: PROGRAM FINISHED SUCCESSFULLY", "magenta"))
