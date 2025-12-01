import pytorch_lightning as pl
import torch
from torch import nn


class GazeTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()
        self.model = net
        self.data_loader = data_loader
        self.criterion = nn.KLDivLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        img = output[0].detach().cpu()
        self.logger.experiment.add_image("predicted_train", img, self.current_epoch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.criterion(output, y)

        self.log("test_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        # Log to tensorboard
        img = output[0].detach().cpu()
        self.logger.experiment.add_image("predicted_test", img, self.current_epoch)

        img = y[0].detach().cpu()
        self.logger.experiment.add_image("ground_truth", img, self.current_epoch)

        img = x[0].detach().cpu()
        self.logger.experiment.add_image("input", img, self.current_epoch)

        return loss

    def train_dataloader(self):
        return self.data_loader["train"]

    def val_dataloader(self):
        return self.data_loader["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2.5 * 1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        return {"optimizer": optimizer}
