import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from models import *
from plotter import *
from general_utils import *
from gym.envs.registration import register
import cv2

register(
    id='SpritesState-v2',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 0}
)

task = 'SpritesState-v2'
env = gym.make(task, render_mode='human')

# input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
input_dim = 4
action_dim = 4

agent = PPOAgent(input_dim, action_dim, lr=0.001, ac_ratio=5)

action_dict = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3:[0, 1]}

num_episodes = 10000
total_rewards = agent.train(env, num_episodes, action_dict)

plot_and_save_loss_per_epoch_1(total_rewards, f'A2C_{task}', 'agents')
cv2.destroyAllWindows()