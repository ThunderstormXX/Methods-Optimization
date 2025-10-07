"""
Небольшая CNN для задачи классификации MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Архитектура:
    - Conv2d(1, 16, 3x3) + ReLU
    - Conv2d(16, 32, 3x3) + ReLU
    - MaxPool2d(2x2)
    - Flatten
    - Linear(4608, 64) + ReLU
    - Linear(64, 10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(4608, 64)  # исправлено: 32 * 12 * 12
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 16, 26, 26)
        x = F.relu(self.conv2(x))  # (batch, 32, 24, 24)
        x = F.max_pool2d(x, 2)     # (batch, 32, 12, 12)
        x = torch.flatten(x, 1)    # (batch, 4608)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
