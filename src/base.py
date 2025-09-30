import lightning as L
import torch
from eye_model import eyegaze_model
from torch import nn


class Light(L.LightningModule):
    def __init__(
        self,
        data,
    ):
        super().__init__()
        self.mod = eyegaze_model()
        # self.optimizer = optium.Adam(modela.parameters(), lr=0.0001)

    def forward(self, x):
        return self.mod(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss()(output, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss()(output, y)

        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
