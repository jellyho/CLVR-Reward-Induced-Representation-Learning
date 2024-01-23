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
from sac import *

parser = argparse.ArgumentParser(description='Reward')
parser.add_argument('-t', '--task', help='Specify the task')
parser.add_argument('-m', '--model', help='Specify the model[oracle, cnn, encoder]')
parser.add_argument('-d', '--dir', help='Specify the save directory')
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
    agent = SAC_CNN(state_dim=(1, 64, 64), action_dim=action_dim)
elif args.model == 'reward_prediction':
    agent = SAC_RewardPrediction(state_dim=(1, 64, 64), action_dim=action_dim)
elif args.model == 'image_scratch':
    agent = SAC_ImageScratch(state_dim=(1, 64, 64), action_dim=action_dim)
elif args.model == 'oracle':
    agent = SAC(input_dim, action_dim)

total_rewards = agent.train(env, render=False, batch_size=256, model_name=args.model, env_name=args.task, save_dir=args.dir)
cv2.destroyAllWindows()