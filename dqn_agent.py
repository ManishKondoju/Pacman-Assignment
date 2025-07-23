import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        assert h >= 32 and w >= 32, "Input must be at least 32x32"

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute conv output dimensions
        def conv2d_out_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_out_size(conv2d_out_size(conv2d_out_size(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_out_size(conv2d_out_size(conv2d_out_size(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Separate streams for value and advantage
        self.fc_value = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # Normalize
        features = self.conv(x).view(x.size(0), -1)
        value = self.fc_value(features)
        advantage = self.fc_advantage(features)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals