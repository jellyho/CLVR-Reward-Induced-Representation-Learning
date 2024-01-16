import numpy as np
import torch
import torch.nn as nn
import datetime
import cv2

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
        # print(x)
        seq = self.lstm(x)
        # print(seq)
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
        x = x[self.N:]
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

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        # Actor 신경망
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)  # 확률로 변환하기 위해 Softmax 사용
        )

        # Critic 신경망
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        # Actor의 출력은 각 행동에 대한 확률 분포
        action_prob = self.actor(state)

        # Critic의 출력은 상태의 가치 (scalar)
        state_value = self.critic(state)

        return action_prob, state_value
    
class A2CAgent:
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, ac_ratio=1):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.model = ActorCritic(input_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.ac_ratio = ac_ratio

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.LongTensor(done)

        action_prob, value = self.model(state)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.gamma * next_value
        critic_loss = (target - value).pow(2).mean()

        eye = torch.eye(self.action_dim)
        one_hot = eye[action]
        adv = (target - value).detach()
        actor_loss = -(torch.log((one_hot * action_prob).sum(1)) * adv).mean()

        # Total loss
        total_loss = self.ac_ratio * actor_loss + critic_loss

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
    def train(self, env, num_episodes=1000, action_dict=None):
        total_rewards = []
        for e in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action_prob, _ = self.model(torch.FloatTensor(state))
                action = np.random.choice(len(action_prob), p=action_prob.detach().numpy())
                if action_dict:
                    next_state, reward, done, _ = env.step(action_dict[action])
                else:
                    next_state, reward, done, _ = env.step(action)

                self.update(state, [action], [reward], next_state, [done])
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"Episode {e}, Total Reward: {total_reward}")
        torch.save(self.model.state_dict(), f'Results/agents/A2C_{datetime.datetime.now()}.pth')
        return total_rewards

class PPOAgent(A2CAgent):
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, ac_ratio=1, epsilon=0.2):
        super(PPOAgent, self).__init__(input_dim, action_dim, lr, gamma, ac_ratio)
        self.epsilon = epsilon

    def update(self, state, action, action_prob_before, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.LongTensor(done)
        action_prob_before = torch.FloatTensor(action_prob_before)

        action_prob, value = self.model(state)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.gamma * next_value
        critic_loss = (target - value).pow(2).mean() # mse for critic

        adv = (target - value).detach()

        ratio = torch.exp(action_prob - action_prob_before)

        surro1 = ratio * adv
        surro2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv

        actor_loss = -torch.min(surro1, surro2).mean()

        loss = critic_loss + self.ac_ratio * actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, env, num_episodes=1000, action_dict=None, verbose=False):
        total_rewards = []
        for e in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                if verbose:
                    img = env.render()
                    cv2.imshow('train', img)
                    cv2.waitKey(1)
                action_prob, _ = self.model(torch.FloatTensor(state))
                action = np.random.choice(len(action_prob), p=action_prob.detach().numpy())
                if action_dict:
                    next_state, reward, done, _ = env.step(action_dict[action])
                else:
                    next_state, reward, done, _ = env.step(action)

                self.update(state, [action], [action_prob[action]], [reward], next_state, [done])
                state = next_state
                total_reward += reward
            total_rewards.append(total_reward)
            print(f"PPO :: Episode {e}, Total Reward: {total_reward}")
        torch.save(self.model.state_dict(), f'Results/agents/PPO_{datetime.datetime.now()}.pth')
        return total_rewards