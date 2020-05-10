from tensorflow import keras
import numpy as np
import pandas as pd
import gym
import json
import gym_forex
from matplotlib import pyplot as plt
from PIL import Image

json_file = json.load(open(r"C:\Users\Melgiri\PycharmProjects\ForEx\settings\settings.json"))
image_dimensions = [json_file["Image Size"][0], json_file["Image Size"][1]]

id = "forex-v0"


class ModelAndEnvironment:
	def __init__(self, input_shape=(300, 300), windows=None, df=pd.read_pickle(r"D:\Data\markets\MT5\CADJPY.pkl")):
		if windows is None:
			windows = [32]
		# Create the bare bones environment, remake to our configurations, then remake the environment
		# @TODO: Make sure this gym environment is not recreated anywhere else
		"[NOTE]: This is the main instance of the learning environment that will be referenced through the whole time"
		self.env = gym.make(id=id)
		self.env._get_observation = self._get_observation
		self.env.remake(input_shape=input_shape, windows=windows, df=df)
		
		observation_space = self.env.observation_space
		action_space = self.env.action_space
		
		# Creating the Neural Network
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
		
		self.model = keras.models.Model(inputs=raw_inputs, outputs=output_layer)
	
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


class NeuralNet1:
	def __init__(self, observation_space=gym.spaces.Dict(), action_space=gym.spaces.Discrete(n=2)):
		inputs = []
		raw_inputs = []
		
		if isinstance(observation_space, gym.spaces.Box):
			observations = {"Input_0": observation_space}
		else:
			observations = observation_space.spaces
		index = 0
		for observation in observations.values():
			input_layer = keras.layers.Input(shape=observation.shape, name="Input_{}".format(str(index)))
			raw_inputs.append(input_layer)
			convolutional_layer1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
			                                           name="Convolutional_1_{}".format(str(index)))(input_layer)
			pooling_layer = keras.layers.MaxPool2D(name="Max_Pooling_1_{}".format(str(index)))(convolutional_layer1)
			convolutional_layer1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
			                                           name="Convolutional_2_{}".format(str(index)))(pooling_layer)
			pooling_layer = keras.layers.MaxPool2D(name="Max_Pooling_2_{}".format(str(index)))(convolutional_layer1)
			flattened_layer = keras.layers.Flatten(name="Flattened_{}".format(str(index)))(pooling_layer)
			hidden_layer_1 = keras.layers.Dense(units=32, activation="relu",
			                                    name="Hidden_Layer_1_{}".format(str(index)))(flattened_layer)
			inputs.append(hidden_layer_1)
			index += 1
		if index != 1:
			concatenated_layer = keras.layers.concatenate(inputs=inputs, name="Concatenated_Layer")
		else:
			concatenated_layer = hidden_layer_1
		hidden_layer_2 = keras.layers.Dense(units=32, activation="relu", name="Hidden_Layer_2")(concatenated_layer)
		hidden_layer_3 = keras.layers.Dense(units=8, activation="relu", name="Hidden_Layer_3")(hidden_layer_2)
		output_layer = keras.layers.Dense(units=action_space.n, activation="softmax", name="Output_Layer")(
			hidden_layer_3)
		
		self.model = keras.models.Model(inputs=raw_inputs, outputs=output_layer)
