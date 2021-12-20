from collections import deque
from numpy import array
from collections import namedtuple
from abc import ABC
from torch.utils.data.dataset import IterableDataset

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self.buffer = deque(maxlen=capacity)
	
	def __len__(self) -> int:
		return len(self.buffer)
	
	def append(self, experience: Experience) -> None:
		self.buffer.append(experience)
	
	def sample(self, batch_size: int) -> Tuple:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
		return array(states), array(actions), array(rewards, dtype=np.float32), array(done, dtype=np.bool), array(next_states)


class RLDataset(IterableDataset, ABC):
	def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
		self.buffer = buffer
		self.sample_size = sample_size
	
	def __iter__(self) -> Tuple:
		states, actions, rewards, done, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(done)):
			yield states[i], actions[i], rewards[i], done[i], new_states[i]