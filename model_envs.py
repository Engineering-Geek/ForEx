from core.Keras_Model_Environment import *


class ModelEnv1(ModelAndEnvironment):
	def __init__(self, input_shape=(300, 300), windows=None, df=pd.read_pickle(r"C:\Users\melgi\Markets\data\CADJPY.pkl")):
		temp_env = gym.make("forex-v0")
		self._create_neural_network = self._create_neural_network
		self._get_observation = temp_env._get_observation
		super(ModelEnv1, self).__init__(input_shape=input_shape, windows=windows, df=df)
	
	def _create_neural_network(self, observation_space=gym.spaces.Dict(), action_space=gym.spaces.Discrete(1)):
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
			conv_layer = keras.layers.Conv2D(256, (5, 5), name="Conv2D_1_{}".format(str(index)))(input_layer)
			max_pool_layer = keras.layers.MaxPool2D((2, 2))(conv_layer)
			flattened_layer = keras.layers.Flatten(name="Flattened_{}".format(str(index)))(max_pool_layer)
			inputs.append(flattened_layer)
			index += 1
		if index != 1:
			concatenated_layer = keras.layers.concatenate(inputs=inputs, name="Concatenated_Layer")
		else:
			concatenated_layer = flattened_layer
		hidden_layer_1 = keras.layers.Dense(units=32, activation="relu", name="Hidden_Layer_1")(concatenated_layer)
		output_layer = keras.layers.Dense(units=action_space.n, activation="softmax", name="Outputs")(hidden_layer_1)
		
		return keras.models.Model(inputs=raw_inputs, outputs=output_layer)
