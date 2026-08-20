import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from utils import atari_to_gym

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# Lightning training module
# ================================================================

CONFIG_PATH = Path("configs/config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

game = config["games"][0]


class ActionTraining(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        net: nn.Module,
        data_loader: dict,
    ):
        super().__init__()

        self.net = net
        self.data_loaders = data_loader

        self.learning_rate = float(hparams.get("learning_rate", 1e-3))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    @staticmethod
    def _prepare_images(images: torch.Tensor) -> torch.Tensor:
        """
        Accept either:

            [batch, 4, 84, 84]

        or:

            [batch, 4, 1, 84, 84]

        ActionNet requires [batch, 4, 84, 84].
        """
        images = images.float()

        if images.ndim == 5 and images.shape[2] == 1:
            images = images.squeeze(2)

        if images.ndim != 4:
            raise ValueError(
                "Training images must have shape "
                "[batch, frames, height, width], but received "
                f"{tuple(images.shape)}."
            )

        if images.shape[1] != 4:
            raise ValueError(
                "ActionNet requires four stacked frames, but received "
                f"shape {tuple(images.shape)}."
            )

        return images

    @staticmethod
    def _prepare_actions(actions: torch.Tensor) -> torch.Tensor:
        """
        Prepare four action targets per sample.

        Input:
            [batch, 4]
            [batch, 4, 1]

        Output:
            [batch, 4] containing Gym action IDs 0-8.
        """
        actions = torch.as_tensor(actions)

        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        if actions.ndim != 2:
            raise ValueError(
                "Expected actions with shape [batch, 4], "
                f"but received {tuple(actions.shape)}."
            )

        if actions.shape[1] != 4:
            raise ValueError(
                "Expected four actions per sample, "
                f"but received shape {tuple(actions.shape)}."
            )

        actions = atari_to_gym(actions)

        return actions.long()

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        images, _, actions = batch

        images = self._prepare_images(images)
        targets = self._prepare_actions(actions)

        logits = self.net(images)

        # Expected:
        # logits:  [batch, 4, 9]
        # targets: [batch, 4]
        if logits.ndim != 3:
            raise ValueError(
                "Expected logits with shape [batch, 4, 9], "
                f"but received {tuple(logits.shape)}."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}."
            )

        if logits.shape[2] != 9:
            raise ValueError(
                f"Expected nine Gym action classes, but received {logits.shape[2]}."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, 9),
            targets.reshape(-1),
        )

        predicted_actions = logits.argmax(dim=2)

        accuracy = (predicted_actions == targets).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return self.data_loaders["train"]

    def val_dataloader(self):
        loader = self.data_loaders.get("val")

        if loader is None:
            return []

        return loader

    def test_dataloader(self):
        loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader
