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