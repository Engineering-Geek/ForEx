from tensorflow import keras
import numpy as np
import pandas as pd
import gym
import json
import gym_anytrading
from matplotlib import pyplot as plt
from PIL import Image
import gym

json_file = json.load(open(r"C:\Users\Melgiri\PycharmProjects\ForEx\settings\settings.json"))
image_dimensions = [json_file["Image Size"][0], json_file["Image Size"][1]]


def PriceFrame_to_ImgArray(prices, env):
	fig = plt.figure(num=0, figsize=(image_dimensions[0] / 80, image_dimensions[1] / 80), dpi=80)
	plt.plot(prices)
	w, h = fig.canvas.get_width_height()
	fig.canvas.draw()
	img_np_array = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
	img_np_array.shape = (w, h, 4)
	img_np_array = np.roll(img_np_array, 3, axis=2)
	w, h, d = img_np_array.shape
	img_np_array = np.array(Image.frombytes("RGBA", (w, h), img_np_array.tostring()).convert("RGB"))
	fig.clf()
	
	return img_np_array


class SimpleNet1:
	def __init__(self):
		self.model = None
	
	def build_model(self, input_size=(30,), nb_actions=2):
		shape = (1,) + input_size
		input_layer = keras.layers.Input(shape=shape)
		flattened_layer = keras.layers.Flatten()(input_layer)
		output_layer = keras.layers.Dense(nb_actions, activation="sigmoid")(flattened_layer)
		self.model = keras.Model(inputs=input_layer, outputs=output_layer)
		return self.model
	
	def specialized_data_funnel(self, env=gym.make("forex-v0")):
		prices = env.df.loc[:, 'Close'].to_numpy()
		
		prices = prices[env.frame_bound[0] - env.window_size:env.frame_bound[1]]
		
		diff = np.insert(np.diff(prices), 0, 0)
		signal_features = np.column_stack((prices, diff))
		
		return prices, signal_features


class NeuralNet2(SimpleNet1):
	def build_model(self, input_size=(30,), nb_actions=2):
		shape = (1,) + input_size
		
		input_layer = keras.layers.Input(shape=shape)
		hidden_layer_1 = keras.layers.Flatten()(input_layer)
		hidden_layer_2 = keras.layers.Dense(64, activation="relu")(hidden_layer_1)
		hidden_layer_3 = keras.layers.Dense(32, activation="relu")(hidden_layer_2)
		output_layer = keras.layers.Dense(nb_actions, activation="sigmoid")(hidden_layer_3)
		
		self.model = keras.Model(inputs=input_layer, outputs=output_layer)
		return self.model


class NeuralNet3:
	def __init__(self, observation_space=gym.spaces.Dict(), action_space=gym.spaces.Discrete(n=2)):
		inputs = []
		raw_inputs = []
		
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
