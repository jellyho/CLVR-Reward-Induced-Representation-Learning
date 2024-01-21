import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import cv2
from torch.distributions import Normal
from torch.distributions import multivariate_normal
from plotter import *

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList()
        num_conv = np.log2(input_size)

        for i in range(1, int(num_conv) + 1):
            self.convs.append(nn.Conv2d(1 if i == 1 else int(2 ** i), int(2 ** (i + 1)), kernel_size=4, stride=2, padding=1))

        self.relu = nn.ReLU()
        flatten_size = int(2 ** (num_conv + 1))

        self.linear1 = nn.Linear(flatten_size, 64)

    def forward(self, x):
        for i, c in enumerate(self.convs):
            x = c(x)
            x = self.relu(x)
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = self.relu(x)
        return x
    
class Head(nn.Module):
    def __init__(self, input_size):
        super(Head, self).__init__()

        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x
    
class Unroller(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Unroller, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = self.relu(out)
        return out

class HiddenStateEncoder(nn.Module):
    def __init__(self, encoder):
        super(HiddenStateEncoder, self).__init__()
        self.encoder = encoder

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.lstm = Unroller(32, 512, 64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x) # (30, 64)
        seq = self.lstm(x)
        return seq
    
class FutureRewardsEstimator(nn.Module):
    def __init__(self, HiddenStateEncoder, N, T, Heads=1):
        super(FutureRewardsEstimator, self).__init__()
        self.hse = HiddenStateEncoder
        self.Heads = Heads
        if Heads == 1:
            self.head = Head(64)
        else:
            self.head = nn.ModuleList([Head(64) for _ in range(Heads)])
        self.N = N
        self.T = T

    def forward(self, x):
        x = self.hse(x)
        x = x[-self.T:]
        if self.Heads == 1:
            x = self.head(x)
            x = x.squeeze(1)
        else:
            xs = [self.head[i](x) for i in range(self.Heads)]
            x = torch.stack(xs)
            x = x.squeeze(2)
        return x
        
class Decoder(nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList()
        num_conv = np.log2(output_size)

        for i in range(int(num_conv), 0, -1):
            self.convs.append(nn.ConvTranspose2d(int(2 ** i), 1 if i == 1 else int(2 ** (i - 1)), kernel_size=4, stride=2, padding=1))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        for i, c in enumerate(self.convs):
            x = c(x)
            if i == len(self.convs) - 1:
                x = self.tanh(x)
            else:
                x = self.relu(x)
        return x
    
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

        self.states[self.idx] = torch.FloatTensor(state)
        self.next_states[self.idx] = torch.FloatTensor(next_state)
        self.actions[self.idx] = torch.FloatTensor(action)
        self.rewards[self.idx] = torch.FloatTensor([reward])
        self.dones[self.idx] = torch.FloatTensor([float(done)])
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = self.states[idxs]
        next_states = self.next_states[idxs]
        rewards = self.rewards[idxs]
        actions = self.actions[idxs]
        dones = self.dones[idxs]
        return states, actions, rewards, next_states, dones

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleCNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, 4, kernel_size=3, stride=2)
        self.cnn2 = nn.Conv2d(4, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        self.dummy_input = torch.randn(1, 1, input_dim, input_dim)
        
        self.linear1 = nn.Linear(self._get_flattend_size(), 64)
        self.linear2 = nn.Linear(64, output_dim)

    def _get_flattend_size(self):
        x = self.cnn1(self.dummy_input)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = nn.Flatten()(x)
        return x.shape[1]

    def forward(self,x):
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = nn.Flatten()(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_count):
        super(MLP, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hiddens = nn.ModuleList()
        for _ in range(hidden_count - 1):
            self.hiddens.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        for i in range(len(self.hiddens) - 1):
            x = self.hiddens[i](x)
            x = self.relu(x)
        x = self.output(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.input = nn.Linear(input_dim, 256)
        self.hidden = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        sigma = nn.Tanh()(self.sigma(x))
        mu = self.mu(x)
        return mu, sigma

class SAC:
    def __init__(self, state_dim, action_dim, latent_dim=64, action_scale=1, encoder=None, freeze_encoder=False, gamma=0.99, alpha=0.2, tau=0.005, log_std_max=2, log_std_min=-5, max_global_step=100000, start_learning=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if isinstance(state_dim, tuple):
            self.latent_dim = latent_dim
        else:
            self.latent_dim = self.state_dim
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.log_std_max=log_std_max
        self.log_std_min=log_std_min
        self.max_global_step = max_global_step
        self.start_learning = start_learning
        
        self.encoder = encoder
        self.freeze_encoder = False

        self.q1 = MLP(self.latent_dim + action_dim, 1, 256, 1)
        self.q2 = MLP(self.latent_dim + action_dim, 1, 256, 1)

        self.q1_target = MLP(self.latent_dim + action_dim, 1, 256, 1)
        self.q2_target = MLP(self.latent_dim + action_dim, 1, 256, 1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy = Actor(self.latent_dim, action_dim)

        self.buffer = ReplayBuffer(100000, self.state_dim, action_dim)

        self.optim_q1 = torch.optim.Adam(self.q1.parameters(), lr=0.001)
        self.optim_q2 = torch.optim.Adam(self.q2.parameters(), lr=0.001)
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        if encoder:
            self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.0003)

        #optimizers needed

    def encode_state(self, state):
        if self.encoder:
            if self.freeze_encoder:
                with torch.no_grad():
                    state = self.encoder(state)
            else:
                state = self.encoder(state)
            return state
        return torch.FloatTensor(state)

    def get_action(self, state):
        state = self.encode_state(state)
        mu, sigma = self.policy(state)
        sigma = torch.clamp(sigma, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(sigma) # variance

        normal = Normal(mu, std)

        sample = normal.rsample()
        action = nn.Tanh()(sample)
        
        log_prob = normal.log_prob(sample)
        log_prob -= torch.log(2 * (1 - action.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob, dim=1)
        
        return action, log_prob

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
                    state = torch.FloatTensor(state)
                    action, _ = self.get_action(state.unsqueeze(0))
            if render:
                print(action)

            next_state, reward, done, _ = env.step(action.detach().numpy())
            next_state = torch.FloatTensor(next_state)
            self.buffer.add(state, action, reward, next_state.unsqueeze(0), done)
            state = next_state
            rewards += reward
        return rewards
    
    def get_q_value(self, model, state, action):
        return model(torch.cat([self.encode_state(state), action], dim=1))
    
    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        ## critic update ##################################
        with torch.no_grad():
            next_action, next_log_prob = self.get_action(next_state)
            next_q1 = self.get_q_value(self.q1_target, next_state, next_action)
            next_q2 = self.get_q_value(self.q2_target, next_state, next_action)
            minq = torch.min(next_q1, next_q2).squeeze(1)
            q_target = reward + self.gamma * (1 - done) * (minq - self.alpha * next_log_prob).detach()

        q1 = self.get_q_value(self.q1, state, action).squeeze(1)
        q2 = self.get_q_value(self.q2, state, action).squeeze(1)

        q1_loss = torch.nn.functional.mse_loss(q1, q_target)
        q2_loss = torch.nn.functional.mse_loss(q2, q_target)

        self.optim_q1.zero_grad()   
        q1_loss.backward()
        self.optim_q1.step()

        self.optim_q2.zero_grad()
        q2_loss.backward()
        self.optim_q2.step()
        ## end of the critic update ###################################

        ## actor update ###############################################
        sampled_action, log_prob = self.get_action(state)
        sampled_q1 = self.get_q_value(self.q1, state, action)
        sampled_q2 = self.get_q_value(self.q2, state, action)
        sampled_minq = torch.min(sampled_q1, sampled_q2).squeeze(1)
        policy_loss = (self.alpha * log_prob - sampled_minq).mean()

        self.optim_policy.zero_grad()
        policy_loss.backward()
        self.optim_policy.step()
        ## end of the actor update
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item()
    
    def update_target(self):
        ## soft update for critic
        for source_params, target_params in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_params.data.copy_((1 - self.tau) * target_params.data + self.tau * source_params.data)
        for source_params, target_params in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_params.data.copy_((1 - self.tau) * target_params.data + self.tau * source_params.data)

    def save_weights(self, dir):
        torch.save(self.policy.state_dict(), dir+'_policy.pth')
        torch.save(self.q1.state_dict(), dir+'_q1.pth')
        torch.save(self.q2.state_dict(), dir+'_q2.pth')

    def load_weights(self, dir):
        self.policy.load_state_dict(torch.load(dir+'_policy.pth'))
        self.q1.load_state_dict(torch.load(dir+'_q1.pth'))
        self.q2.load_state_dict(torch.load(dir+'_q2.pth'))

    def train(self, env, render=False, batch_size=256, name='oracle'):
        total_rewards = []
        q1 = []
        q2 = []
        policy = []

        for e in range(self.max_global_step):
            if e < self.start_learning:
                # rollout for fill buffer
                self.rollout(env, random=True)
            else:
                total_rewards.append(self.rollout(env, render=render))
                q1_loss, q2_loss, policy_loss = self.update(batch_size)
                q1.append(q1_loss)
                q2.append(q2_loss)
                policy.append(policy_loss)

                self.update_target()

                if (e + 1) % 100 == 0:
                    print(f"Episode {e + 1}, Total Reward: {np.mean(total_rewards[-50:])}, Q1-{np.mean(q1[-10:])}, Q2-{np.mean(q2[-10:])}, Po-{np.mean(policy[-10:])}")
                if (e + 1) % 5000 == 0:
                    self.save_weights(f'Results/agents/SAC_{name}_{e + 1}')
                    np.save(f'{name}.npy', np.array(total_rewards))
        return total_rewards
        


