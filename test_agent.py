import torch
import gym
from models import *
from plotter import *
from general_utils import *
from gym.envs.registration import register
import cv2
import time
import argparse
from sac import *

parser = argparse.ArgumentParser(description='Reward')
parser.add_argument('-t', '--task', help='Specify the task')
parser.add_argument('-m', '--model', help='Specify the model[oracle, cnn, encoder]')
parser.add_argument('-d', '--dir', help='Specify the save directory')
parser.add_argument('-e', '--epoch', help='Specify the model[oracle, cnn, encoder]')
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
action_dim = 2

if args.model == 'cnn':
    agent = SAC_CNN(state_dim=(1, 64, 64), action_dim=action_dim)
elif args.model == 'encoder':
    agent = SAC_Encoder(state_dim=(1, 64, 64), action_dim=action_dim)
elif args.model == 'oracle':
    agent = SAC(input_dim, action_dim)
    agent.load_weights(f'{args.dir}/{args.task}_{args.model}_{args.epoch}')


while True:
    agent.rollout(env, random=False, render=True, fps=10)