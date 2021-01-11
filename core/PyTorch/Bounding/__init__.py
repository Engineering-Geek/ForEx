from abc import ABC

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from typing import List
import pandas as pd

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
		batch = batch.float()
		
		input_layer = []
		for index in range(len(batch[0, 0, :])):
			input_layer_index = self.input(batch[:, :, index])
			input_layer.append(input_layer_index)
		
		middle_layer = self.middle_layer(torch.cat(input_layer, 1))
		
		mean_prediction = self.mean_prediction(middle_layer)
		std_prediction = self.mean_prediction(middle_layer)
		
		output = torch.cat([mean_prediction, std_prediction])
		return output
	
	def training_step(self, batch, batch_idx):
		data, target = batch
		logits = self.forward(data)
		loss = F.nll_loss(logits, target)
		return {'loss': loss}
	
	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=1e-3)
	
	def train_dataloader(self) -> DataLoader:
		df = pd.read_pickle(file_path)
		# This operation will take up A SHIT TON of memory, if memory on RAM runs out, it will be likely here
		raw_data = df.to_numpy()
		final_data = []
		for window in self.windows:
			for index in range(len(raw_data)):
				pass


def train():
	obs_size = 128
	trainer = Trainer()
	model = BoundingAI(obs_size)
