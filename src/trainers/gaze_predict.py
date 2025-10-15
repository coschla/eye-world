import pytorch_lightning as pl


class AuxiliaryTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        pass

    def forward(self, x, command):
        pass

    def training_step(self, batch, batch_idx):
        # TODO: 1. Add training logic and logging logic

        raise NotImplementedError  # loss

    def validation_step(self, batch, batch_idx):
        # TODO: 1. Add training logic and logging logic

        raise NotImplementedError  # loss

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def configure_optimizers(self):
        # TODO: Add optimizer configuration
        pass
