import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.conv_jepa import ConvEncoder
from models.jepa import encoder_frames


class ConvJEPA(pl.LightningModule):
    """
    Action-conditioned conv JEPA: from the latent of frames t-3..t and the `horizon`
    actions taken from t onwards, predict the EMA-teacher latent of the frames `horizon`
    steps later. Loss: L1 on layer-normed latents + a variance hinge on the student
    latent across the batch (keeps samples apart, so no collapse).
    """

    def __init__(self, config, dim=512, ctx=4, horizon=4, num_actions=18, act_dim=32, lr=3e-4, ema=0.99, reg_coeff=1.0):
        super().__init__()
        self.config, self.ctx, self.horizon = config, ctx, horizon
        self.lr, self.ema, self.reg_coeff = lr, ema, reg_coeff
        self.encoder = ConvEncoder(ctx, dim)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.act_embed = torch.nn.Embedding(num_actions, act_dim)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(dim + horizon * act_dim, 2 * dim), torch.nn.GELU(), torch.nn.Linear(2 * dim, dim)
        )

    def predict(self, z, actions):
        return self.pred(torch.cat([z, self.act_embed(actions).flatten(1)], dim=-1))

    def on_train_batch_end(self, *args):
        with torch.no_grad():
            for s, t in zip(self.encoder.parameters(), self.target.parameters()):
                t.mul_(self.ema).add_(s, alpha=1 - self.ema)

    def training_step(self, batch, batch_idx):
        img, _, actions = batch  # frames [B, ctx + horizon, H, W]; actions [B, ctx + horizon]
        frames = encoder_frames(img, self.ctx + self.horizon, self.config.get("grey_scale", True)).squeeze(2)
        context = frames[:, : self.ctx]
        future = frames[:, self.horizon : self.horizon + self.ctx]
        taken = actions[:, self.ctx - 1 : self.ctx - 1 + self.horizon].long()  # a_t .. a_{t+horizon-1}

        z = self.encoder(context)
        z = F.layer_norm(z, (z.size(-1),))
        with torch.no_grad():
            tgt = self.target(future)
            tgt = F.layer_norm(tgt, (tgt.size(-1),))

        loss_jepa = F.l1_loss(self.predict(z, taken), tgt)
        between = z.std(dim=0)
        loss_reg = F.relu(1.0 - between).mean()
        loss = loss_jepa + self.reg_coeff * loss_reg

        self.log("loss/jepa", loss_jepa, on_epoch=True, prog_bar=True)
        self.log("loss/reg", loss_reg, on_epoch=True)
        # Does the predictor beat copying the current latent as its guess for the future?
        self.log("loss/copy_baseline", F.l1_loss(z.detach(), tgt), on_epoch=True, prog_bar=True)
        self.log("rep/between_std", between.mean(), on_epoch=True)
        with torch.no_grad():  # does the predictor use the actions? (equal to loss/jepa = it ignores them)
            wrong = F.l1_loss(self.predict(z, taken.roll(1, dims=0)), tgt)
        self.log("loss/jepa_wrong_actions", wrong, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.act_embed.parameters()) + list(self.pred.parameters()), lr=self.lr, weight_decay=1e-4
        )
