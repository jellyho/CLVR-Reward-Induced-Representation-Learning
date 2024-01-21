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
import argparse

parser = argparse.ArgumentParser(description='Reward')
parser.add_argument('-t', '--task', help='Specify the task')
parser.add_argument('-r', '--reward', help='Specify the reward')
parser.add_argument('-m', '--model', help='Specify the model[oracle, cnn, encoder]')
args = parser.parse_args()

#### Image-based follower envs. ####
register(
    id='Sprites-v0',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='Sprites-v1',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 1}
)

register(
    id='Sprites-v2',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 2}
)

#### State-based follower envs. ####
register(
    id='SpritesState-v0',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='SpritesState-v1',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 1}
)


register(
    id='SpritesState-v2',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 2}
)

task = args.task
env = gym.make(task, render_mode='human')

input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
action_dim = 2

if args.model == 'cnn':
    input_dim = (1, 64, 64)
    encoder = SimpleCNN(64, 64)
    agent = SAC(input_dim, action_dim, encoder=encoder, freeze_encoder=False, start_learning=1000, max_global_step=50000)
elif args.model == 'encoder':
    input_dim = (1, 64, 64)
    encoder = Encoder(64)
    encoder.load_state_dict(torch.load(f'./Results/encoder/encoder_six.pth'))
    input_dim = 64
    agent = SAC(input_dim, action_dim, encoder=encoder, freeze_encoder=True)
elif args.model == 'oracle':
    encoder = None
    agent = SAC(input_dim, action_dim, latent_dim=4, encoder=encoder)

total_rewards = agent.train(env, render=False, batch_size=256, name=args.task)
cv2.destroyAllWindows()