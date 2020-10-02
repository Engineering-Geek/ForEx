# ForEx
This project's primary purpose is to take my savings from co-op/internships and invest them in various markets. This ForEx trading bot will then trade on various platforms and amplify my earnings, and hopefully generate enough cash flow for me to start my own startup.

## Core workings
This bot will consist of X neural networks, where X numbers is being experimented with. I am testing various code inputs ranging from convolutional neural networks to standard pct change inputs. 

I strongly encourage you to look through my code as this documentation will remain incomplete until I finish the code and have it working. What is here is a dramatic oversimplification of what I'm doing. I have more in-depth documentation within the code.

For example, in the convolutional input layer, the bot will see this:

![graph image](https://i.imgur.com/03KsW34.png)

and in the traditional percent change input, the model will see something similar to this (just the last column; column E):

![table image](https://i.stack.imgur.com/wvmET.png)

I am currently working on automating the testing procedures so the model can test various models via Reinforcement learning and show me a visualization of the statistics with either Matplotlib or with any other visualizing program.
