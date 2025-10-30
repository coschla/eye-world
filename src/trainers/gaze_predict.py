<<<<<<< HEAD
import pytorch_lightning as pl
=======
import pytorch_lightning as L
>>>>>>> dd199f7 (frame visulization created)
import torch
from torch import nn


class GazeTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()
        self.model = net
        self.data_loader = data_loader
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("test_data", loss, on_epoch=True, on_step=False)
        return loss

    def train_dataloader(self):
        return self.data_loader["train"]

    def val_dataloader(self):
        return self.data_loader["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
<<<<<<< HEAD
        return {"optimizer": optimizer}
=======
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "test_data",  # required for this scheduler
            },
        }
        # TODO: Add optimizer configuration
>>>>>>> dd199f7 (frame visulization created)
