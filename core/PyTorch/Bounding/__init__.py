from abc import ABC

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as functional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from typing import List
import pandas as pd
import numpy as np

file_path = "C:/Users/melgi/Markets/data/EURUSD.pkl"


class BoundingAI(LightningModule):
	def __init__(self, obs_size: List[List[int]], windows: List[int]):
		super().__init__()
		self.windows = windows
		# Model Declaration
		self.input_layer = nn.Sequential(
			nn.Linear(obs_size[0], 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.Softmax()
		)
		self.middle_layer = nn.Sequential(
			nn.Linear(32 * 4, 32),
			nn.ReLU()
		)
		self.mean_prediction = nn.Sequential(
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)
		self.std_prediction = nn.Sequential(
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)
	
	def forward(self, batch: Tensor):
		input_layer = []
		for index in range(len(batch[0, 0, :])):
			input_layer_index = self.input_layer(batch[:, :, index])
			input_layer.append(input_layer_index)
		
		middle_layer = self.middle_layer(torch.cat(input_layer, 1))
		
		mean_prediction = self.mean_prediction(middle_layer)
		std_prediction = self.mean_prediction(middle_layer)
		
		output = torch.cat([mean_prediction, std_prediction], dim=1)
		return output
	
	def training_step(self, input_batch, batch_idx):
		output = self.forward(batch=input_batch)
		mean = output[:, 0]
		std_dev = output[:, 0]
		upper_bound = mean + std_dev
		lower_bound = mean + std_dev
		
		return {'loss': 0}
	
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-3)
	
	def train_dataloader(self) -> DataLoader:
		dataloader = DataLoader(dataset=ForexDataset(path=file_path, windows=self.windows), batch_size=128)
		return dataloader


class ForexDataset(Dataset):
	def __init__(self, path: str, windows: List[int], target_distance: int):
		super().__init__()
		self.data = pd.read_pickle(path).to_numpy()
		self.max_window = max(windows)
		self.windows = windows
		self.target_distance = target_distance

	def __len__(self):
		return len(self.data) - self.max_window

	def __getitem__(self, index):
		index += self.max_window
		x = []
		for window in self.windows:
			x.append(self.data[index - window: index])
		if len(self.windows) == 1:
			x = x[0]
		# Get
		return torch.Tensor(x)


def train():
	obs_size = 128
	trainer = Trainer()
	model = BoundingAI([128, 4], [128])
	trainer.fit(model)


if __name__ == '__main__':
	train()
