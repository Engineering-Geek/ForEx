from tensorflow import keras
import gym
import gym_anytrading


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
		start = env.frame_bound[0] - env.window_size
		end = env.frame_bound[1]
		prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
		signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]
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
