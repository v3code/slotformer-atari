import pickle
import numpy as np
import torch
import torch.nn as nn
import gym
import slotformer.rl.envs.shapes

env = gym.make("Navigation5x5-v0")
print(env.action_space)


