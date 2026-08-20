import sys
from pathlib import Path

import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

# from src.utils skip_run

# ================================================================
# Configuration
# ================================================================

CONFIG_PATH = Path("configs/config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

game = config["games"][0]
# ================================================================
# Action network
# ================================================================


class ActionNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=8,
                stride=4,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # An 84x84 input produces 64 x 7 x 7 = 3136 features.
        self.num_actions = num_actions
        self.num_predictions = 4

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                self.num_predictions * self.num_actions,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        # [batch, 36] -> [batch, 4, 9]
        x = x.reshape(
            x.shape[0],
            self.num_predictions,
            self.num_actions,
        )

        return x
