import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.utils import block_mask_tubelets_vectorized
from trainers.utils import format_batch_for_vjepa

config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


class VJEPA(pl.LightningModule):
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

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def _rep_stats(self, x):
        """x: [B, N, D]"""
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T, H, W]
        B = img.shape[0]

        # --------------------------------------------------
        # Tokenize once — shared by student and teacher
        # --------------------------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.model.tubelet_embed(x)  # [B, N, D]
        N, D = all_tokens.shape[1], all_tokens.shape[2]

        # --------------------------------------------------
        # Block masking — uniform N_masked across batch
        # --------------------------------------------------
        _, mask_bool = block_mask_tubelets_vectorized(
            all_tokens,
            drop_ratio=self.mask_ratio,
            block_size=self.config.get("block_size", 2),
        )  # mask_bool: [B, N], True = masked

        if mask_bool.sum() == 0:
            return torch.tensor(0.0, device=img.device, requires_grad=True)

        # --------------------------------------------------
        # Teacher — sees all tokens, no grad
        # --------------------------------------------------
        with torch.no_grad():
            target_full = self.teacher(all_tokens)  # [B, N, D]

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
        # Loss: smooth-L1 + variance regularization
        # --------------------------------------------------
        loss_jepa = F.smooth_l1_loss(pred, target_masked)
        pred_std = pred.std(dim=1)
        loss_reg = F.relu(1.0 - pred_std).mean()
        loss = loss_jepa + self.reg_coeff * loss_reg

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target_masked)

        self.log("loss/jepa", loss_jepa, on_epoch=True, prog_bar=True)
        self.log("loss/reg", loss_reg, on_epoch=True)
        self.log("loss/total", loss, on_epoch=True, prog_bar=True)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)

        with torch.no_grad():
            n_params = sum(1 for _ in self.model.student.parameters())
            drift = sum(
                (s - t).pow(2).mean()
                for s, t in zip(
                    self.model.student.parameters(), self.teacher.parameters()
                )
            )
            self.log("ema/drift", drift / n_params, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.pred.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )


class ActionConditionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        action_embed,
        config,
        latent_pred_dim=None,
        num_visible_frames=4,
        lr=1e-4,
        ema_decay=0.996,
    ):
        """
        Args:
            model: VJEPAEncoder containing student + tubelet_embed
            config: dict, must contain "action_dim"
            ckpt_path: path to pretrained V-JEPA checkpoint
            latent_pred_dim: optional dim for latent predictor
            num_visible_frames: how many frames student sees
        """
        super().__init__()
        self.model = model
        self.config = config
        self.lr = lr
        self.ema_decay = ema_decay
        self.num_visible_frames = num_visible_frames

        # Teacher starts as copy of student
        self.teacher = copy.deepcopy(model.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Action embedding
        self.action_embed = action_embed

        # Autoregressive latent predictor (causal transformer)
        D = model.student.encoder.layers[0].self_attn.embed_dim
        latent_pred_dim = latent_pred_dim or D
        self.latent_predictor = nn.Transformer(
            d_model=D,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )
        """
        ckpt_path = config["ckpt_path"]
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # Load weights
            student_weights = {
                k.replace("model.student.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.student.")
            }
            self.model.student.load_state_dict(student_weights)

            # Load tubelet embedding weights
            embed_weights = {
                k.replace("model.tubelet_embed.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.tubelet_embed.")
            }
            self.model.tubelet_embed.load_state_dict(embed_weights)
        except FileNotFoundError:
            pass"""

        # Teacher starts as EMA of student
        self.teacher.load_state_dict(self.model.student.state_dict())

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):
        # -------------------------------
        # Format batch
        # -------------------------------
        img, actions = format_batch_for_vjepa(batch, self.config)

        if isinstance(img, list):
            img = torch.stack(img)

        context_frames = self.config[
            "context_frames"
        ]  # n frames for student; rest go to teacher
        # -------------------------------
        # Split sequence (CRITICAL)
        # -------------------------------
        student_frames = img[:, :context_frames]
        teacher_frame = img[:, context_frames:]
        # teacher_frame = img[:, 1:5]
        student_x = student_frames.unsqueeze(2)  # [B, 4, 1, H, W]
        teacher_x = teacher_frame.unsqueeze(2)  # [B, 1, 1, H, W]

        # -------------------------------
        # Student forward
        # -------------------------------
        with torch.no_grad():
            student_tokens = self.model.tubelet_embed(student_x)  # [B, N, D]
            student_tokens = self.model.student.encoder(student_tokens)

        # -------------------------------
        # Action embedding
        # -------------------------------
        last_actions = actions[:, context_frames - 1]
        action_emb = self.action_embed(last_actions)
        action_emb = action_emb.unsqueeze(1)

        # -------------------------------
        # Combine tokens + action
        # -------------------------------
        seq = torch.cat([student_tokens, action_emb], dim=1)  # [B, N+1, D]

        # BUG: drop last token before prediction
        pred_seq = self.latent_predictor(seq[:, :-1], seq[:, :-1])

        student_pred = pred_seq[:, -1, :]

        """
        # -------------------------------
        # Action embedding (FIXED)
        # -------------------------------
        last_actions = actions[
            :, context_frames - 1
        ]  # action at last context frame [B]
        action_emb = self.action_embed(last_actions)  # [B, D]
        action_emb = action_emb.unsqueeze(1)  # [B, 1, D]







        # -------------------------------
        # Combine tokens + action
        # -------------------------------
        seq = torch.cat([student_tokens, action_emb], dim=1)  # [B, N+1, D]
            """

        # -------------------------------
        # Teacher forward (FIXED)
        # -------------------------------
        with torch.no_grad():
            teacher_tokens = self.model.tubelet_embed(teacher_x)  # [B, Nt, D]
            teacher_latents = self.teacher(teacher_tokens)  # [B, Nt, D]

        # -------------------------------
        # Predictor
        # -------------------------------
        # Use sequence except last token as input
        pred_seq = self.latent_predictor(seq[:, :-1, :], seq[:, :-1, :])

        # Take last predicted token

        pred_seq = self.latent_predictor(seq, seq)

        student_pred = pred_seq[:, -1, :]

        # student_pred = pred_seq[:, -1, :]  # [B, D]

        # -------------------------------
        # Target
        # -------------------------------
        target = teacher_latents.mean(dim=1)  # [B, D]

        # -------------------------------
        # Loss
        # -------------------------------
        loss = F.smooth_l1_loss(student_pred, target)

        self.log("loss/action_jepa", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )


import pytorch_lightning as pl


class RolloutActionVJEPA(pl.LightningModule):
    """
    3rd loop:
    Multi-step action-conditioned JEPA rollout training.

    Key idea:
    z_{t+1} = f(z_t, a_t)

    BUT applied recursively for K steps:
    z_{t+k} = f(f(...f(z_t, a_t), a_{t+1})..., a_{t+k})
    """

    def __init__(
        self,
        model,
        action_embed,
        config,
        latent_predictor,
        rollout_steps=8,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()

        self.model = model
        self.action_embed = action_embed
        self.config = config
        self.rollout_steps = rollout_steps
        self.lr = lr
        self.ema_decay = ema_decay

        # frozen encoder (from V-JEPA stage 1/2)
        self.encoder = model.student
        for p in self.encoder.parameters():
            p.requires_grad = False

        # teacher (EMA encoder for targets)
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # transition model (your ActionCondition predictor)
        self.latent_predictor = latent_predictor

    # ----------------------------
    # EMA update
    # ----------------------------
    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.encoder.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    # ----------------------------
    # one-step transition
    # ----------------------------
    def predict_next(self, latent, action):
        """
        latent: [B, N, D]
        action: [B]
        """

        a = self.action_embed(action).unsqueeze(1)  # [B,1,D]

        seq = torch.cat([latent, a], dim=1)  # [B, N+1, D]

        out = self.latent_predictor(seq, seq)

        # remove action token effect
        return out[:, :-1, :]

    # ----------------------------
    # training
    # ----------------------------
    def training_step(self, batch, batch_idx):

        img, actions = batch
        B = img.shape[0]

        context_frames = self.config["context_frames"]

        # ----------------------------
        # encode initial state
        # ----------------------------
        context = img[:, :context_frames].unsqueeze(2)
        future = img[:, context_frames : context_frames + self.rollout_steps]

        with torch.no_grad():
            z = self.model.tubelet_embed(context)
            z = self.encoder(z)

        # ----------------------------
        # teacher rollout targets
        # ----------------------------
        targets = []

        with torch.no_grad():
            for t in range(self.rollout_steps):
                frame = future[:, t].unsqueeze(1).unsqueeze(2)
                z_t = self.model.tubelet_embed(frame)
                z_t = self.teacher(z_t)
                targets.append(z_t)

        # ----------------------------
        # student rollout
        # ----------------------------
        preds = []

        latent = z

        for t in range(self.rollout_steps):
            a_t = actions[:, context_frames + t]

            latent = self.predict_next(latent, a_t)

            preds.append(latent)

        # ----------------------------
        # rollout loss (temporal masking style)
        # ----------------------------
        loss = 0.0

        for p, g in zip(preds, targets):
            loss += F.smooth_l1_loss(p, g)

        loss = loss / self.rollout_steps

        self.log("loss/rollout_vjepa", loss, on_epoch=True, prog_bar=True)

        return loss

    # ----------------------------
    # optimizer
    # ----------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )


"""


def training_step(self, batch, batch_idx):

    # --------------------------------------------------
    # Format batch
    # --------------------------------------------------
    img, actions, gaze_locations = format_batch_for_vjepa(
        batch,
        self.config,
    )

    if isinstance(img, list):
        img = torch.stack(img)

    context_frames = self.config["context_frames"]

    student_frames = img[:, :context_frames]
    teacher_frames = img[:, context_frames:]

    student_x = student_frames.unsqueeze(2)
    teacher_x = teacher_frames.unsqueeze(2)

    # --------------------------------------------------
    # Student encoder
    # --------------------------------------------------
    student_tokens = self.model.tubelet_embed(student_x)
    student_tokens = self.model.student(student_tokens)

    # --------------------------------------------------
    # Teacher encoder (EMA)
    # --------------------------------------------------
    with torch.no_grad():
        teacher_tokens = self.model.tubelet_embed(teacher_x)
        teacher_latents = self.teacher(teacher_tokens)

    # --------------------------------------------------
    # Compute gaze token indices (optional)
    # --------------------------------------------------
    gaze_indices = None

    if (
        self.config["use_gaze_for_predictor"]
        or
        self.config["use_gaze_for_loss"]
    ):

        heatmap = eye_gaze_to_density_image(
            student_frames.shape,
            gaze_locations,
            self.config,
        )

        gaze_indices = self.compute_gaze_indices(heatmap)

    # --------------------------------------------------
    # Predictor input
    # --------------------------------------------------
    predictor_tokens = student_tokens

    if (
        self.config["use_gaze_for_predictor"]
        and gaze_indices is not None
    ):
        predictor_tokens = self.gather_tokens(
            predictor_tokens,
            gaze_indices,
        )

    # --------------------------------------------------
    # Action token
    # --------------------------------------------------
    last_action = actions[:, context_frames - 1]

    action_token = self.action_embed(last_action)
    action_token = action_token.unsqueeze(1)

    predictor_input = torch.cat(
        [
            predictor_tokens,
            action_token,
        ],
        dim=1,
    )

    # --------------------------------------------------
    # Predict future tubelets
    # --------------------------------------------------
    prediction = self.latent_predictor(
        predictor_input,
        predictor_input,
    )

    # Remove the action token prediction
    prediction = prediction[:, :-1]

    # --------------------------------------------------
    # Build target
    # --------------------------------------------------
    target = teacher_latents

    if gaze_indices is not None:

        target = self.gather_tokens(
            target,
            gaze_indices,
        )

    # --------------------------------------------------
    # Compute loss
    # --------------------------------------------------
    loss = F.smooth_l1_loss(
        prediction,
        target,
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    self.log(
        "loss/action_jepa",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )

    return loss

"""


"""

class ActionConditionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        action_embed,
        config,
        latent_pred_dim=None,
        num_visible_frames=4,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()

        self.model = model
        self.config = config
        self.lr = lr
        self.ema_decay = ema_decay
        self.num_visible_frames = num_visible_frames

        # --------------------------------------------------
        # Load pretrained V-JEPA checkpoint
        # --------------------------------------------------
        ckpt_path = config["ckpt_path"]

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")

            student_weights = {
                k.replace("model.student.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.student.")
            }
            self.model.student.load_state_dict(student_weights)

            embed_weights = {
                k.replace("model.tubelet_embed.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.tubelet_embed.")
            }
            self.model.tubelet_embed.load_state_dict(embed_weights)

            print(f"Loaded pretrained checkpoint: {ckpt_path}")

        except FileNotFoundError:
            print("No pretrained checkpoint found.")

        # --------------------------------------------------
        # Freeze pretrained encoder
        # --------------------------------------------------
        for p in self.model.student.parameters():
            p.requires_grad = False

        for p in self.model.tubelet_embed.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # Teacher (EMA copy)
        # --------------------------------------------------
        self.teacher = copy.deepcopy(self.model.student)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher.load_state_dict(self.model.student.state_dict())

        # --------------------------------------------------
        # Action embedding
        # --------------------------------------------------
        self.action_embed = action_embed

        # --------------------------------------------------
        # Latent predictor
        # --------------------------------------------------
        D = model.student.encoder.layers[0].self_attn.embed_dim
        latent_pred_dim = latent_pred_dim or D

        self.latent_predictor = nn.Transformer(
            d_model=D,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(
                self.model.student.parameters(),
                self.teacher.parameters(),
            ):
                t.data.mul_(self.ema_decay).add_(
                    s.data,
                    alpha=1.0 - self.ema_decay,
                )

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):

        # --------------------------------------------------
        # Format batch
        # --------------------------------------------------
        img, actions, gaze_locations = format_batch_for_vjepa(
            batch,
            self.config,
        )

        if isinstance(img, list):
            img = torch.stack(img)

        context_frames = self.config["context_frames"]

        student_frames = img[:, :context_frames]
        teacher_frames = img[:, context_frames:]

        student_x = student_frames.unsqueeze(2)
        teacher_x = teacher_frames.unsqueeze(2)

        # --------------------------------------------------
        # Student encoder (frozen)
        # --------------------------------------------------
        with torch.no_grad():
            student_tokens = self.model.tubelet_embed(student_x)
            student_tokens = self.model.student(student_tokens)

        # --------------------------------------------------
        # Teacher encoder
        # --------------------------------------------------
        with torch.no_grad():
            teacher_tokens = self.model.tubelet_embed(teacher_x)
            teacher_latents = self.teacher(teacher_tokens)

        # --------------------------------------------------
        # Eye gaze processing
        # --------------------------------------------------
        gaze_indices = None

        if (
            self.config["use_gaze_for_predictor"]
            or self.config["use_gaze_for_loss"]
        ):

            heatmap = eye_gaze_to_density_image(
                student_frames.shape,
                gaze_locations,
                self.config,
            )

            gaze_indices = self.compute_gaze_indices(
                heatmap
            )

        # --------------------------------------------------
        # Predictor input
        # --------------------------------------------------
        predictor_tokens = student_tokens

        if (
            self.config["use_gaze_for_predictor"]
            and gaze_indices is not None
        ):
            predictor_tokens = self.gather_tokens(
                predictor_tokens,
                gaze_indices,
            )

        # --------------------------------------------------
        # Action token
        # --------------------------------------------------
        last_action = actions[:, context_frames - 1]

        action_token = self.action_embed(
            last_action
        ).unsqueeze(1)

        predictor_input = torch.cat(
            [
                predictor_tokens,
                action_token,
            ],
            dim=1,
        )

        # --------------------------------------------------
        # Predict future latents
        # --------------------------------------------------
        prediction = self.latent_predictor(
            predictor_input,
            predictor_input,
        )

        # Remove prediction corresponding to action token
        prediction = prediction[:, :-1]

        # --------------------------------------------------
        # Build target
        # --------------------------------------------------
        target = teacher_latents

        if (
            self.config["use_gaze_for_loss"]
            and gaze_indices is not None
        ):
            target = self.gather_tokens(
                target,
                gaze_indices,
            )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        loss = F.smooth_l1_loss(
            prediction,
            target,
        )

        self.log(
            "loss/action_jepa",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
"""
