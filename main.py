# Importing custom classes and functions
from core.dataframe_processor import *
from core.Training import *
from core.Neural_Networks import *

from gym_anytrading.envs import ForexEnv
from gym import spaces

from termcolor import colored

# select which parent model to use
model_parent = Single_Input_Convolutional


# Specialized environment for gym as specified by the parent model above. Edit according to the neural
#   network you are using.
class SpecializedForexEnv(ForexEnv):
	_process_data = model_parent.specialized_data_funnel
	_get_observation = model_parent.specialized_observation

	def __init__(self, df=pd.DataFrame(), window_size=(), frame_bound=(), unit_side="left"):
		super().__init__(
			df=df,
			window_size=window_size,
			frame_bound=frame_bound,
			unit_side=unit_side
		)
		self.shape = self.signal_features.shape
		self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)


# ------------------------------------------------[MAIN]------------------------------------------------
print(colored("[INFO]: Generating Data for environment", "green"))
# data = DataProcessor(
# 	pickle_path=r"D:\Data\markets\CADJPY.pkl"
# )
data = pd.read_pickle(r"/media/melgiri/Storage/Data/markets/CADJPY_3.pkl")

print(colored("[INFO]: Creating ForEx environment for training", "green"))
env = SpecializedForexEnv(
	df=data,
	window_size=12,
	frame_bound=(12, len(data))
)

print(colored("[INFO]: Building Neural Network", "green"))
model = model_parent().build_model(
	input_size=env.observation_space.shape,
	nb_actions=env.action_space.n
)
print(colored("[INFO]: OBS Space- {}".format(str(env.observation_space)), "green"))

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
