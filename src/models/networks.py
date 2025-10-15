import pytorch_lightning as pl


class ConvNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet, self).__init__()

    # TODO: Implement the network architecture here

    def forward(self, img):
        # TODO: Impelement the forward propogation
        raise NotImplementedError
