import torch
import gym
from models import *
from plotter import *
from general_utils import *
from gym.envs.registration import register
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='Reward')
parser.add_argument('-t', '--task', help='Specify the task')
parser.add_argument('-r', '--root', help='Specify the reward')
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
action_dim = 2

if args.model == 'cnn':
    input_dim = (1, 64, 64)
    encoder = SimpleCNN(64, 64)
    agent = SAC(input_dim, action_dim, encoder=encoder, freeze_encoder=False)
elif args.model == 'encoder':
    input_dim = (1, 64, 64)
    encoder = Encoder(64)
    encoder.load_state_dict(torch.load(f'./Results/encoder/encoder_six.pth'))
    agent = SAC(input_dim, action_dim, latent_dim=64, encoder=encoder, freeze_encoder=True)
elif args.model == 'oracle':
    encoder = False
    agent = SAC(input_dim, action_dim, encoder=encoder)
    agent.load_weights(f'{args.root}/{args.model}_73000')


while True:
    agent.rollout(env, random=False, render=True, fps=10)