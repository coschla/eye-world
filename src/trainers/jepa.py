import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.jepa import encoder_frames
from models.utils import block_mask_tubelets_vectorized


class JEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        pred,
        config,
        mask_ratio=0.5,
        lr=1e-4,
        ema_decay=0.996,
        reg_coeff=0.1,
    ):
        super().__init__()
        self.model = model
        self.teacher = copy.deepcopy(model.student)
        self.pred = pred
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay
        self.reg_coeff = reg_coeff

        for p in self.teacher.parameters():
            p.requires_grad = False

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    # TODO: confirm this hook fires. on_after_optimizer_step is not a
    # LightningModule hook in PL 2.x (use on_train_batch_end or
    # on_before_optimizer_step); if it never fires the teacher stays at its
    # random init. Check that the ema/drift log changes during training.
    # TODO: V-JEPA 2 uses a momentum schedule (about 0.998 -> 1.0), not a
    # fixed 0.996, and EMAs the patch embedding together with the encoder
    # (here tubelet_embed is shared and not EMA'd).
    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def _rep_stats(self, x):
        """x: [B, N, D]"""
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T*C, H, W] stacked frames
        B = img.shape[0]

        # --------------------------------------------------
        # Tokenize once — shared by student and teacher
        # --------------------------------------------------
        x = encoder_frames(
            img, int(self.config["stack_length"]), self.config.get("grey_scale", True)
        )  # [B, T, C, H, W]
        all_tokens = self.model.tubelet_embed(x)  # [B, N, D]
        N, D = all_tokens.shape[1], all_tokens.shape[2]

        # --------------------------------------------------
        # Block masking — uniform N_masked across batch
        # --------------------------------------------------
        _, mask_bool = block_mask_tubelets_vectorized(
            all_tokens,
            drop_ratio=self.mask_ratio,
            # TODO: block_size=2 masks two neighbouring patches, which
            # is easy to fill in by interpolation. V-JEPA 2 masks large
            # contiguous spatiotemporal blocks (multi-block masking).
            block_size=self.config.get("block_size", 2),
        )  # mask_bool: [B, N], True = masked

        if mask_bool.sum() == 0:
            return torch.tensor(0.0, device=img.device, requires_grad=True)

        # --------------------------------------------------
        # Teacher — sees all tokens, no grad
        # --------------------------------------------------
        with torch.no_grad():
            # Layer-norm each target token over its features (no learnable
            # affine), as V-JEPA 2 does, so the target scale is fixed.
            target_full = self.teacher(all_tokens)  # [B, N, D]
            target_full = F.layer_norm(target_full, (D,))

        # --------------------------------------------------
        # Student — sees only visible tokens
        # --------------------------------------------------
        N_visible = int((~mask_bool[0]).sum())
        visible_tokens = all_tokens[~mask_bool].reshape(B, N_visible, D)
        student_repr = self.model.student(visible_tokens)  # [B, N_visible, D]

        # --------------------------------------------------
        # Predictor
        # --------------------------------------------------
        N_masked = N - N_visible
        pos_embed = self.model.tubelet_embed.pos_embed.expand(B, -1, -1)
        masked_pos = pos_embed[mask_bool].reshape(B, N_masked, D)
        target_masked = target_full[mask_bool].reshape(B, N_masked, D)

        pred = self.pred(queries=masked_pos, context=student_repr)

        # --------------------------------------------------
        # Loss: L1 + variance regularization
        # --------------------------------------------------
        loss_jepa = F.l1_loss(pred, target_masked)
        # TODO: the targets are now unit-scale per token, so the hinge below
        # (std over tokens within a sample, threshold 1.0) may push the
        # prediction away from the targets. Compare rep/pred_var with
        # rep/tgt_var; if the targets themselves have std < 1 across tokens,
        # lower the threshold or set reg_coeff to 0 (V-JEPA 2 has no such term).
        pred_std = pred.std(dim=1)
        loss_reg = F.relu(1.0 - pred_std).mean()
        loss = loss_jepa + self.reg_coeff * loss_reg

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target_masked)

        self.log("loss/jepa", loss_jepa, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss/reg", loss_reg, on_epoch=True, sync_dist=True)
        self.log("loss/total", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True, sync_dist=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True, sync_dist=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True, sync_dist=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True, sync_dist=True)

        with torch.no_grad():
            n_params = sum(1 for _ in self.model.student.parameters())
            drift = sum(
                (s - t).pow(2).mean()
                for s, t in zip(
                    self.model.student.parameters(), self.teacher.parameters()
                )
            )
            self.log("ema/drift", drift / n_params, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.pred.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
