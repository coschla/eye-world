from pathlib import Path

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """The imitation CNN's conv trunk, ending in a latent vector. [B, T, 84, 84] -> [B, dim]."""

    def __init__(self, in_channels: int = 4, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, dim),
        )

    def forward(self, x):
        return self.net(x)


class ConvActionNet(nn.Module):
    """ConvEncoder (optionally pretrained and frozen) + MLP action head. [B, 4, 84, 84] -> [B, 1, 18]."""

    def __init__(self, num_actions, in_channels=4, dim=512, jepa_checkpoint=None, freeze_encoder=True):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, dim)
        if jepa_checkpoint is not None:
            self._load(Path(jepa_checkpoint))
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def _load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Conv JEPA checkpoint was not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        # Pretraining module keeps the student under "encoder." (the EMA copy is "target.")
        state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(state, strict=True)
        print("Loaded pretrained conv JEPA encoder:", path)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, images):
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder(images)
        else:
            z = self.encoder(images)
        return self.head(z).unsqueeze(1)
