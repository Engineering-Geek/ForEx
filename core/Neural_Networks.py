from tensorflow import keras
import numpy as np
import pandas as pd
import gym
import json
import gym_anytrading
from matplotlib import pyplot as plt
from PIL import Image

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


class NeuralNet3(SimpleNet1):
	def specialized_data_funnel(self, env=gym.make("forex-v0")):
		prices_df = env.df.loc[:, 'Close']
		
		prices = prices_df.to_numpy()[env.frame_bound[0] - env.window_size:env.frame_bound[1]]
		# Just create a single image for the library to understand the observation space. Don't
		#   worry about the data input for now
		img_np_array = np.asarray(PriceFrame_to_ImgArray(prices_df.to_numpy(), env))
		
		return prices, img_np_array
	
	def specialized_observation(self, env=gym.make("forex-v0")):
		# Use the price-frame_to_image-array function to generate series of images and return a single image
		if env._current_tick is None:
			env._current_tick = env.window_size
		img_array = PriceFrame_to_ImgArray(
			prices=env.prices[env._current_tick - env.window_size: env._current_tick],
			env=env
		)
		return img_array
