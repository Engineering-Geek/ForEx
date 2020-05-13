from tensorflow import keras
import numpy as np
import pandas as pd
import gym
import json
import gym_forex

json_file = json.load(open(r"D:\Data\markets\settings\settings.json"))
image_dimensions = [json_file["Image Size"][0], json_file["Image Size"][1]]

env_name = "forex-v0"
base_env = gym.make(env_name)


class ModelAndEnvironment:
	def __init__(self, input_shape=(300, 300), windows=None, df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl")):
		if windows is None:
			windows = [32]
		# -------------------------------------------------[CREATE ENV]-------------------------------------------------
		"[NOTE]: This is the main instance of the learning environment that will be referenced through the whole time"
		self.env = base_env
		self.env._get_observation = self._get_observation
		self.env.__init__(input_shape=input_shape, windows=windows, df=df)
		
		self.model = self._create_neural_network(
			observation_space=self.env.observation_space,
			action_space=self.env.action_space
		)
	
	def _create_neural_network(self, observation_space=gym.spaces.Box(0, 1, (1,)), action_space=gym.spaces.Discrete(1)):
		raw_inputs = []
		inputs = []
		index = 0
		if isinstance(observation_space, gym.spaces.Box):
			observations = {"Input_0": observation_space}
		else:
			observations = observation_space.spaces
		
		for observation in observations.values():
			input_layer = keras.layers.Input(shape=observation.shape, name="Input_{}".format(str(index)))
			raw_inputs.append(input_layer)
			flattened_layer = keras.layers.Flatten(name="Flattened_{}".format(str(index)))(input_layer)
			inputs.append(flattened_layer)
			index += 1
		if index != 1:
			concatenated_layer = keras.layers.concatenate(inputs=inputs, name="Concatenated_Layer")
		else:
			concatenated_layer = flattened_layer
		hidden_layer_1 = keras.layers.Dense(units=32, activation="relu", name="Hidden_Layer_1")(concatenated_layer)
		hidden_layer_2 = keras.layers.Dense(units=8, activation="relu", name="Hidden_Layer_3")(hidden_layer_1)
		output_layer = keras.layers.Dense(units=action_space.n, activation="softmax", name="Outputs")(hidden_layer_2)
		
		return keras.models.Model(inputs=raw_inputs, outputs=output_layer)
	
	def _get_observation(self):
		index = self.env.index
		windows = self.env.windows
		if self.env.custom_df is None:
			self.env.custom_df = self.env.df.pct_change()
		
		observations = []
		for window in windows:
			custom_window = self.env.custom_df.iloc[index - window: index]
			observations.append(np.asarray(custom_window))
		
		return observations


# These are examples of using the class above to create a new observation, only here as an example format
# ----------------------------------------------[DO NOT CALL OR USE THESE]----------------------------------------------

# Example 1
class ModelAndEnvironmentBlankExample(ModelAndEnvironment):
	def __init__(self, input_shape=(300, 300), windows=None, df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl")):
		super(ModelAndEnvironmentBlankExample, self).__init__(input_shape=input_shape, windows=windows, df=df)
	
	def _create_neural_network(self, observation_space=gym.spaces.Box(0, 1, (1,)), action_space=gym.spaces.Discrete(1)):
		pass
	
	def _get_observation(self):
		pass


# Example 2
def create_neural_net():
	return keras.Sequential()


class ModelAndEnvironmentExample2(ModelAndEnvironment):
	def __init__(self, input_shape=(300, 300), windows=None, df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl")):
		self._get_observation = base_env._get_observation
		self._create_neural_network = create_neural_net
		super(ModelAndEnvironmentExample2, self).__init__(input_shape=input_shape, windows=windows, df=df)
