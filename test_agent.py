import torch
import gym
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
env = gym.make(task)
input_dim = 4
action_dim = 2
# 
# agent = A2CAgent(input_dim, action_dim)
agent = A2CAgent(input_dim, action_dim)
agent.network.load_state_dict(torch.load(f'./Results/agents/SAC.pth'))

while True:
    state = env.reset()
    done = False
    while not done:
        # env.render()
        img = env.render()
        cv2.imshow('train', img)
        cv2.waitKey(1)

        # action_prob, _ = agent.model(torch.FloatTensor(state))
        action = agent.get_action(torch.FloatTensor(state))
        # action = np.random.choice(len(action_prob), p=action_prob.detach().numpy())
        next_state, reward, done, _ = env.step(action.detach().numpy())
        state = next_state