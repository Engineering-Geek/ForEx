from gym.envs.registration import register
import pandas as pd

register(
	id='forex-v0',
	entry_point='gym_forex.envs:ForexEnv',
	kwargs={
		"input_shape": (299, 299, 3),
		"nb_actions": 2,
		"windows": None,
		"df": pd.DataFrame(),
	}
)
