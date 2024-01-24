import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import cv2
from torch.distributions import Normal
from torch.distributions import multivariate_normal
from plotter import *
from models import SimpleCNN, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer: #FIFO
    def __init__(self, max_len, state_dim, action_dim):
        self.max_len = max_len
        self.idx = 0
        self.size = 0

        if isinstance(state_dim, tuple):
            self.states = torch.zeros((max_len,) + state_dim, dtype=torch.float32)
            self.next_states = torch.zeros((max_len,) + state_dim, dtype=torch.float32)
        elif isinstance(state_dim, int):
            self.states = torch.zeros((max_len, state_dim), dtype=torch.float32)
            self.next_states = torch.zeros((max_len, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros(max_len, dtype=torch.float32)
        self.actions = torch.zeros((max_len, action_dim), dtype=torch.float32)
        self.dones = torch.zeros(max_len, dtype=torch.float32)

    def add(self, state, action, reward, next_state, done):
        self.idx = (self.idx + 1) % self.max_len

        self.states[self.idx] = torch.FloatTensor(np.array(state))
        self.next_states[self.idx] = torch.FloatTensor(np.array(next_state))
        self.actions[self.idx] = torch.FloatTensor(np.array(action))
        self.rewards[self.idx] = torch.FloatTensor(np.array([reward]))
        self.dones[self.idx] = torch.FloatTensor(np.array([float(done)]))
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = self.states[idxs]
        next_states = self.next_states[idxs]
        rewards = self.rewards[idxs]
        actions = self.actions[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.input = nn.Linear(input_dim, 256)
        self.hidden = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        sigma = self.sigma(x)
        sigma = nn.Tanh()(sigma)
        sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (sigma + 1)
        mu = self.mu(x)
        return mu, sigma
    
    def get_action(self, state):
        mu, sigma = self(state)
        std = torch.exp(sigma) # variance

        normal = Normal(mu, std)

        sample = normal.rsample()
        action = nn.Tanh()(sample)
        log_prob = normal.log_prob(sample)

        log_prob -= torch.log(2 * (1 - action.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob, dim=1)

        return action, log_prob
    
class Actor_CNN(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim):
        super(Actor_CNN, self).__init__()
        self.cnn = SimpleCNN(input_dim, latent_dim)
        self.input = nn.Linear(latent_dim, 256)
        self.hidden = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.cnn(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        sigma = self.sigma(x)
        sigma = nn.Tanh()(sigma)
        sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (sigma + 1)
        mu = self.mu(x)
        return mu, sigma
    
    def get_action(self, state):
        mu, sigma = self(state)
        std = torch.exp(sigma) # variance

        normal = Normal(mu, std)

        sample = normal.rsample()
        action = nn.Tanh()(sample)
        log_prob = normal.log_prob(sample)

        log_prob -= torch.log(2 * (1 - action.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob, dim=1)

        return action, log_prob
    
class Actor_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim, encoder_weight_dir=None):
        super(Actor_Encoder, self).__init__()
        self.encoder = Encoder(input_dim)
        self.freeze = False if encoder_weight_dir is None else True
        if encoder_weight_dir is not None:
            self.encoder.load_state_dict(torch.load(encoder_weight_dir))
        self.input = nn.Linear(latent_dim, 256)
        self.hidden = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        sigma = self.sigma(x)
        sigma = nn.Tanh()(sigma)
        sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (sigma + 1)
        mu = self.mu(x)
        return mu, sigma
    
    def get_action(self, state):
        mu, sigma = self(state)
        std = torch.exp(sigma) # variance

        normal = Normal(mu, std)

        sample = normal.rsample()
        action = nn.Tanh()(sample)
        log_prob = normal.log_prob(sample)

        log_prob -= torch.log(2 * (1 - action.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob, dim=1)

        return action, log_prob
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class QNetwork_CNN(nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim):
        super().__init__()
        self.cnn = SimpleCNN(state_dim, latent_dim)
        self.fc1 = nn.Linear(latent_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        x = self.cnn(x)
        x = torch.cat([x, a], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class QNetwork_Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim, encoder_weight_dir=None):
        super().__init__()
        self.encoder = Encoder(state_dim)
        self.freeze = False if encoder_weight_dir is None else True
        if encoder_weight_dir is not None:
            self.encoder.load_state_dict(torch.load(encoder_weight_dir))
        self.fc1 = nn.Linear(latent_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        if self.freeze:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        x = torch.cat([x, a], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SAC:
    def __init__(self, state_dim, action_dim, action_scale=1, gamma=0.99, alpha=0.2, tau=0.005, max_global_step=35000, start_learning=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.max_global_step = max_global_step
        self.start_learning = start_learning

        self.init_networks()

        self.buffer = ReplayBuffer(50000, self.state_dim, action_dim)

    def init_networks(self):
        self.q1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q2 = QNetwork(self.state_dim, self.action_dim).to(device)

        self.q1_target = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q2_target = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.optim_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=0.001)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    def rollout(self, env, random=False, render=False, fps=1000):
        rewards = 0
        state = env.reset()
        done = False
        while not done:
            if render:
                img = env.render()
                cv2.imshow('train', img)
                cv2.waitKey(int(1000 / fps))
            if random:
                action = torch.FloatTensor(np.random.uniform(-1, 1, 2))
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(np.array(state))
                    action, _ = self.actor.get_action(state.unsqueeze(0).to(device))
            if render:
                print(action)

            next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
            next_state = torch.FloatTensor(np.array(next_state))
            self.buffer.add(state, action.cpu(), reward, next_state.unsqueeze(0), done)
            state = next_state
            rewards += reward
        return rewards

    def train(self, env, render=False, batch_size=256, model_name='oracle', env_name='', save_dir=''):
        total_rewards = []
        q_losses = []
        actor_losses = []
        
        for e in range(self.max_global_step):
            if e < self.start_learning:
                self.rollout(env, random=True)
            else:
                total_rewards.append(self.rollout(env, render=render))

                # updating q network
                state, action, reward, next_state, done = self.buffer.sample(batch_size)
                state = state.to(device)
                action = action.to(device)
                reward = reward.to(device)
                next_state = next_state.to(device)
                done = done.to(device)
                with torch.no_grad():
                    next_action, next_log_prob = self.actor.get_action(next_state)
                    next_q1 = self.q1_target(next_state, next_action)
                    next_q2 = self.q2_target(next_state, next_action)
                    q_target = torch.min(next_q1, next_q2).squeeze(1) - self.alpha * next_log_prob
                    next_q_value = reward + self.gamma * (1 - done) * q_target

                q1 = self.q1(state, action).squeeze(1)
                q2 = self.q2(state, action).squeeze(1)

                q1_loss = torch.nn.functional.mse_loss(q1, next_q_value)
                q2_loss = torch.nn.functional.mse_loss(q2, next_q_value)

                q_loss = q1_loss + q2_loss

                self.optim_q.zero_grad()
                q_loss.backward()
                self.optim_q.step()

                # updating actor

                sampled_action, sampled_log_prob = self.actor.get_action(state)
                sampled_q1 = self.q1(state, sampled_action)
                sampled_q2 = self.q2(state, sampled_action)

                sampled_min_q = torch.min(sampled_q1, sampled_q2)
                actor_loss = (self.alpha * sampled_log_prob - sampled_min_q).mean()

                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step()

                q_losses.append(q_loss.item())
                actor_losses.append(actor_loss.item())

                self.update_target()
                if (e + 1) % 100 == 0:
                    print(f"Episode {e + 1}, Total Reward: {np.mean(total_rewards[-50:])}, QLoss-{np.mean(q_losses[-50:])}, ALoss-{np.mean(actor_losses[-50:])}")
                if (e + 1) % 5000 == 0:
                    self.save_weights(f'{save_dir}/{env_name}_{model_name}_{e + 1}')
                    np.save(f'{save_dir}/{env_name}_{model_name}_Log.npy', np.array(total_rewards))
        return total_rewards

    def update_target(self):
        ## soft update for critic
        for source_params, target_params in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_params.data.copy_((1 - self.tau) * target_params.data + self.tau * source_params.data)
        for source_params, target_params in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_params.data.copy_((1 - self.tau) * target_params.data + self.tau * source_params.data)

    def save_weights(self, dir):
        torch.save(self.actor.state_dict(), dir+'_actor.pth')
        torch.save(self.q1.state_dict(), dir+'_q1.pth')
        torch.save(self.q2.state_dict(), dir+'_q2.pth')

    def load_weights(self, dir):
        self.actor.load_state_dict(torch.load(dir+'_actor.pth', map_location=device))
        self.q1.load_state_dict(torch.load(dir+'_q1.pth', map_location=device))
        self.q2.load_state_dict(torch.load(dir+'_q2.pth', map_location=device))

class SAC_CNN(SAC):
    def init_networks(self):
        self.q1 = QNetwork_CNN(64, 64, self.action_dim).to(device)
        self.q2 = QNetwork_CNN(64, 64, self.action_dim).to(device)

        self.q1_target = QNetwork_CNN(64, 64, self.action_dim).to(device)
        self.q2_target = QNetwork_CNN(64, 64, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor = Actor_CNN(64, 64, self.action_dim).to(device)
        self.optim_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=0.001)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

class SAC_ImageScratch(SAC):
    def init_networks(self):
        self.q1 = QNetwork_Encoder(64, 64, self.action_dim).to(device)
        self.q2 = QNetwork_Encoder(64, 64, self.action_dim).to(device)

        self.q1_target = QNetwork_Encoder(64, 64, self.action_dim).to(device)
        self.q2_target = QNetwork_Encoder(64, 64, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor = Actor_Encoder(64, 64, self.action_dim).to(device)
        self.optim_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=0.001)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

class SAC_RewardPrediction(SAC):
    def init_networks(self):
        self.q1 = QNetwork_Encoder(64, 64, self.action_dim, f'./Results/encoder/encoder_six.pth').to(device)
        self.q2 = QNetwork_Encoder(64, 64, self.action_dim, f'./Results/encoder/encoder_six.pth').to(device)

        self.q1_target = QNetwork_Encoder(64, 64, self.action_dim, f'./Results/encoder/encoder_six.pth').to(device)
        self.q2_target = QNetwork_Encoder(64, 64, self.action_dim, f'./Results/encoder/encoder_six.pth').to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor = Actor_Encoder(64, 64, self.action_dim, './Results/encoder/encoder_six.pth').to(device)
        self.optim_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=0.001)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)