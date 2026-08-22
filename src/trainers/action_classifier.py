import pytorch_lightning as pl
import torch
from torch import nn

from models.utils import atari_to_gym

# ================================================================
# Lightning training module
# ================================================================


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
        self.criterion = nn.CrossEntropyLoss()

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

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, _, actions = batch

        images = images.float()

        if images.ndim == 5:
            images = images.squeeze(2)
        targets = atari_to_gym(torch.as_tensor(actions).squeeze(-1)).long()

        logits = self.net(images)

        loss = self.criterion(
            logits.reshape(-1, 9),
            targets.reshape(-1),
        )

        predicted_actions = logits.argmax(dim=2)
        accuracy = (predicted_actions == targets).float().mean()

        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, prog_bar=True)

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
