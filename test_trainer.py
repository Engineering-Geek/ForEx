import argparse

import pytorch_lightning as pl

from core.Pytorch_Implimentation_2 import DQNLightning


def main(hparams) -> None:
	model = DQNLightning(hparams)
	trainer = pl.Trainer(
		gpus=1,
		distributed_backend="dp",
		max_epochs=5000,
		val_check_interval=1000
	)
	trainer.fit(model)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--env", type=str, default="forex-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
parser.add_argument("--warm_start_size", type=int, default=1000, help="samples to fill buffer at the start of training")
parser.add_argument("--eps_last_frame", type=int, default=10000, help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=128*4, help="max length of an episode")
parser.add_argument("--max_episode_reward", type=int, default=200, help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000, help="max episode reward in the environment")

if __name__ == '__main__':
	args, _ = parser.parse_known_args()
	main(args)
