import torch
import torch.nn as nn


class ActionNet(nn.Module):
    def __init__(self, num_actions):
        super(ActionNet, self).__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # Atari style input
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # Fully connected action head
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),  # 3136
            nn.LeakyReLU(),
            nn.Linear(512, num_actions),
        )  # 1568

    def forward(self, x):
        x = self.conv_layers(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        x = self.fc_layers(x)

        return x  # logits


"""


class ActionNet(nn.Module):
    def __init__(self, num_actions, num_future=2):
        super(ActionNet, self).__init__()

        self.num_future = num_future
        self.num_actions = num_actions

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),  # ✅ fixed bug
            nn.LeakyReLU(),
        )

        # Fully connected head
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_actions * num_future),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.fc_layers(x)
        logits = logits.view(-1, self.num_future, self.num_actions)  # [B, 2, A]

        return logits
"""
