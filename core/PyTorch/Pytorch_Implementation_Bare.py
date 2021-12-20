from pl_bolts.models.rl.dqn_model import DQN
from pytorch_lightning import Trainer

model = DQN("CartPole-v1")
trainer = Trainer()
trainer.fit(model)
