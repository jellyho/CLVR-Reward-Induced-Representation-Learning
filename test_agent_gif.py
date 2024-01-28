import torch
import gym
from models import *
from plotter import *
from general_utils import *
from gym.envs.registration import register
import cv2
from sac import *

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

input_dim = 4
action_dim = 2
base_dir = './Results/agents'
tasks = [
    ['SpritesState-v0'] + ['Sprites-v0'] * 3,
    ['SpritesState-v1'] + ['Sprites-v1'] * 3,
    ['SpritesState-v2'] + ['Sprites-v2'] * 3
]
baselines = ['oracle', 'cnn', 'image_scratch', 'reward_predictor']
epochs = [
    [50000, 30000, 35000, 35000],
    [20000, 30000, 35000, 30000],
    [40000, 35000, 35000, 25000]
]

total_rollouts = 3
fps = 10
images = np.ones((3, 4, total_rollouts * 40, 1, 64, 64))

for env_idx in range(3):
    for task_idx in range(4):
        env = gym.make(tasks[env_idx][task_idx], render_mode='human')

        input_dim = env.observation_space.shape[0]
        action_dim = 2
        # to reduce the memory usage
        if task_idx == 0:
            agent = SAC(input_dim, action_dim)
        elif task_idx == 1:
            agent = SAC_CNN(state_dim=(1, 64, 64), action_dim=action_dim)
        elif task_idx == 2:
            agent = SAC_ImageScratch(state_dim=(1, 64, 64), action_dim=action_dim)
        elif task_idx == 3:
            agent = SAC_RewardPrediction(state_dim=(1, 64, 64), action_dim=action_dim)
        agent.load_weights(f'{base_dir}/{tasks[env_idx][task_idx]}_{baselines[task_idx]}_{epochs[env_idx][task_idx]}')

        tmp_stack = []
        for _ in range(total_rollouts):
            _, stack = agent.rollout(env, random=False, render=True, fps=1000, stack=True)
            tmp_stack.append(stack)
        tmp_stack = np.concatenate(tmp_stack)
        images[env_idx, task_idx, :, :, :, :] = tmp_stack

        del agent

width = (64 * 4 + 10 * 8) * 2
height = 120 * 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 1
font_color = 0
for env_idx in range(3):
    image = np.ones((40 * total_rollouts, height, width)) * 255
    for frame in range(40 * total_rollouts):
        x = 10 * 2
        y = 60
        for target_idx in range(4):
            image[frame, y:y + (64 * 2), x:x + (64 * 2)] = cv2.resize(images[env_idx, target_idx, frame, :, :].squeeze(0), None, fy=2, fx=2)
            text_size = cv2.getTextSize(baselines[target_idx], font, font_scale / 2, font_thickness)[0]
            cv2.putText(image[frame, :, :], baselines[target_idx], (int((x + 64 - text_size[0] / 2)), y + 64 * 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale / 2, font_color, font_thickness, cv2.LINE_AA)
            x += 64 * 2
            x += 20 * 2
        # put text and title
        text_size = cv2.getTextSize(tasks[env_idx][1], font, font_scale, font_thickness * 2)[0]
        cv2.putText(image[frame, :, :], tasks[env_idx][1], (int(((width - text_size[0]) / 2)), 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness * 2, cv2.LINE_AA)
    create_gif(image, f'./Results/{tasks[env_idx][1]}.gif', 5)