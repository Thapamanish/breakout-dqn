#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))
        self.advantage = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, num_actions))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.max().detach() > 1.5:
            x = x.float().div_(255.0)
        x = self.features(x).flatten(1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy layer used for exploration in Rainbow DQN."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size, device):
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scaled_noise(self.in_features, self.weight_mu.device)
        epsilon_out = self._scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """Dueling C51 DQN with NoisyNet exploration.

    ``forward`` returns expected action values for action selection; ``dist``
    returns the categorical return distribution required by the C51 loss.
    """

    def __init__(self, in_channels=4, num_actions=4, num_atoms=51,
                 v_min=-10.0, v_max=10.0):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        hidden = 512
        self.value_hidden = NoisyLinear(64 * 7 * 7, hidden)
        self.value_out = NoisyLinear(hidden, self.num_atoms)
        self.advantage_hidden = NoisyLinear(64 * 7 * 7, hidden)
        self.advantage_out = NoisyLinear(hidden, self.num_actions * self.num_atoms)
        self.register_buffer('support', torch.linspace(self.v_min, self.v_max, self.num_atoms))
        self._init_convolutions()

    def _init_convolutions(self):
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.constant_(module.bias, 0.0)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def _features(self, x):
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.max().detach() > 1.5:
            x = x.float().div_(255.0)
        return self.features(x).flatten(1)

    def dist(self, x):
        features = self._features(x)
        value = self.value_out(F.relu(self.value_hidden(features))).view(-1, 1, self.num_atoms)
        advantage = self.advantage_out(F.relu(self.advantage_hidden(features))).view(
            -1, self.num_actions, self.num_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1)

    def forward(self, x):
        probabilities = self.dist(x)
        return torch.sum(probabilities * self.support, dim=-1)
