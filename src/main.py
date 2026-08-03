import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, IterableDataset

from data.data_write import create_webdataset
from dataset.pre_process import ComposePreprocessor, Resize, Stack, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from models.networks import ConvNet, UNet
from models.vjepa import (
    ActionEmbedding,
    Predictor,
    TransformerEncoder,
    TubeletEmbedding,
    VJEPAEncoder,
)
from trainers.gaze_predict import GazeTraining
from trainers.jepa import VJEPA, ActionConditionVJEPA
from trainers.utils import atari_to_gym
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "data_cleaning") as check, check():
    for game in config["games"]:
        create_webdataset(game, config)


with skip_run("skip", "torch_dataset") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )

    for x, y in train_test_dataloaders["train"]:
        print(x.shape)
        print(y.shape)


with skip_run("skip", "gaze_visualization") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    plt.ion()
    fig, ax = plt.subplots()
    for batch_idx, (imgs, labels) in enumerate(train_test_dataloaders["train"]):
        for i in range(len(imgs)):
            img = imgs[i]
            label = labels[i]
            img_np = img.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.set_title(f"Frame {batch_idx},{i} Lable : {label}")
            plt.pause(0.1)
            ax.clear()
    plt.ioff()
    plt.show()


with skip_run("skip", "gaze_prediction") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/gaze_prediction/")
    # gaze prediction network
    net = ConvNet(config=config)

    # Dataloader
    preprocessor = ComposePreprocessor([Resize(config)])
    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    model = GazeTraining(config, net)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


with skip_run("skip", "gaze_prediction_conv_deconv") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/gaze_prediction/")
    # Gaze prediction network
    net = UNet(config=config)

    # Dataloader
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    model = GazeTraining(config, net, train_test_dataloaders)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        devices=[0],
        accelerator="gpu",
        enable_progress_bar=True,
    )
    trainer.fit(model)


with skip_run("skip", "jepa_training_multi_game") as check, check():
    logger = TensorBoardLogger(
        "tb_logs",
        name="multi_game/vjepa_world_model/",
    )

    preprocessor = ComposePreprocessor(
        [
            Resize(config),
            Stack(config),
        ]
    )

    # --------------------------------------------------
    # Load all game datasets
    # --------------------------------------------------
    datasets = []
    example_loader = None

    for game in config["games"]:
        ds = get_torch_dataloaders(
            game,
            config,
            preprocessor=preprocessor,
        )

        datasets.append(ds["train"].dataset)
        train_loader = ds["train"]

        if example_loader is None:
            example_loader = ds["train"]

        # num_samples = sum(1 for _ in train_loader.dataset)
        # print(f"{game}: ~{num_samples} samples (estimated)")

        print(f"Loaded dataset for {game}")

    # --------------------------------------------------
    # Interleave samples from multiple games
    # --------------------------------------------------
    class InterleavedDataset(IterableDataset):
        def __init__(self, datasets):
            self.datasets = datasets

        def __iter__(self):
            iterators = [iter(ds) for ds in self.datasets]

            while iterators:
                idx = random.randrange(len(iterators))

                try:
                    yield next(iterators[idx])

                except StopIteration:
                    iterators.pop(idx)

    combined_dataset = InterleavedDataset(datasets)

    # --------------------------------------------------
    # Build train dataloader
    # --------------------------------------------------
    train_loader = DataLoader(
        combined_dataset,
        batch_size=example_loader.batch_size,
        shuffle=False,
        num_workers=example_loader.num_workers,
        pin_memory=getattr(example_loader, "pin_memory", False),
    )

    # --------------------------------------------------
    # Verify batch shape
    # --------------------------------------------------
    for x, y in train_loader:
        print("Train batch shape:", x.shape)
        print("Label shape:", y.shape)
        break

    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------
    patch_dim = 1 if config.get("grey_scale", True) else 3

    embed_dim = 768
    heads = 12
    mlp_dim = 3072

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )

    student = TransformerEncoder(
        embed_dim,
        depth=12,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    net = VJEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    pred = Predictor(
        embed_dim,
        depth=4,
        heads=heads // 2,
        mlp_dim=mlp_dim,
    )

    model = VJEPA(
        model=net,
        pred=pred,
        config=config,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    trainer.fit(model, train_loader)


with skip_run("skip", "jepa_trainers") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_action_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = 768  # 1024
    heads = 12
    mlp_dim = 3072  # 2048

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )
    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)
    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)
    action_embed = ActionEmbedding()
    model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",  # replaces 'gpus'
        devices=1,  # replaces 'gpus=2'
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, dataloaders["train"])


with skip_run("run", "jepa_rollout_trainer_with_validation") as check, check():
    game = config["games"][0]

    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_rollout_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    # -----------------------------
    # MODEL (same as your 2nd loop)
    # -----------------------------
    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = 768
    heads = 12
    mlp_dim = 3072

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )

    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)

    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)

    action_embed = ActionEmbedding()

    # reuse your existing ActionConditionVJEPA internals
    base_model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    rollout_steps = config.get("rollout_steps", 12)

    # -----------------------------
    # TRAINING LOOP (ROLLOUT VERSION)
    # -----------------------------
    class RolloutActionJEPA(pl.LightningModule):
        def __init__(self, base_model, rollout_steps):
            super().__init__()

            self.model = base_model.model
            self.action_embed = base_model.action_embed
            self.latent_predictor = base_model.latent_predictor

            self.rollout_steps = rollout_steps

            # Freeze everything except predictor
            self.model.tubelet_embed.requires_grad_(False)
            self.model.student.requires_grad_(False)
            self.action_embed.requires_grad_(False)

        # -----------------------------
        # One-step transition
        # -----------------------------
        def predict_next(self, latent, action):
            with torch.no_grad():
                a = self.action_embed(atari_to_gym(action))

                if a.ndim > 2:
                    a = a.mean(dim=tuple(range(1, a.ndim - 1)))

                a = a.unsqueeze(1)
                seq = torch.cat([latent, a], dim=1)

            out = self.latent_predictor(seq, seq)
            return out[:, :-1, :]

        # -----------------------------
        # Shared rollout/loss
        # -----------------------------
        def _shared_step(self, batch):
            img, actions, *_ = batch
            context_frames = config["context_frames"]

            # Encode initial context
            with torch.no_grad():
                context = img[:, :context_frames].unsqueeze(2)
                z = self.model.tubelet_embed(context)
                z = self.model.student(z)

            # Teacher rollout targets
            future = img[:, context_frames : context_frames + self.rollout_steps]

            targets = []

            with torch.no_grad():
                context_window = img[:, :context_frames]

                for t in range(self.rollout_steps):
                    next_frame = future[:, t].unsqueeze(1)

                    context_window = torch.cat(
                        [context_window[:, 1:], next_frame],
                        dim=1,
                    )

                    z_t = self.model.tubelet_embed(context_window.unsqueeze(2))
                    z_t = self.model.student(z_t)

                    targets.append(z_t)

            # Predictor rollout
            preds = []
            latent = z

            for t in range(self.rollout_steps):
                a_t = actions[:, context_frames + t]
                latent = self.predict_next(latent, a_t)
                preds.append(latent)

            loss = sum(F.smooth_l1_loss(p, g) for p, g in zip(preds, targets))
            loss /= self.rollout_steps

            return loss

        # -----------------------------
        # Training
        # -----------------------------
        def training_step(self, batch, batch_idx):
            loss = self._shared_step(batch)

            self.log(
                "train/rollout_dynamics",
                loss,
                on_epoch=True,
                prog_bar=True,
            )

            return loss

        # -----------------------------
        # Validation
        # -----------------------------
        def validation_step(self, batch, batch_idx):
            loss = self._shared_step(batch)

            self.log(
                "val/rollout_dynamics",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            return loss

        # -----------------------------
        # Optimizer
        # -----------------------------
        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.latent_predictor.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )

    # -----------------------------
    # TRAINER
    # -----------------------------
    model = RolloutActionJEPA(base_model, rollout_steps)

    # -----------------------------
    # LOAD PREVIOUS CHECKPOINT
    # -----------------------------
    ckpt = torch.load(
        "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_0/checkpoints/epoch=49-step=164350.ckpt",
        map_location="cpu",
    )

    state_dict = ckpt["state_dict"]

    # Remove obsolete teacher weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print(dataloaders.keys())
    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


# ============================================================
# Latent Rollout
# ============================================================


class LatentRollout(nn.Module):
    """
    Rolls the JEPA latent forward using the learned
    latent dynamics model.

    Produces the future latent after cycle_steps.
    """

    def __init__(
        self,
        latent_predictor,
        action_embed,
    ):
        super().__init__()

        self.latent_predictor = latent_predictor
        self.action_embed = action_embed

    def step(self, latent, action):

        with torch.no_grad():
            action = self.action_embed(action)

            if action.ndim > 2:
                action = action.mean(dim=tuple(range(1, action.ndim - 1)))

            action = action.unsqueeze(1)

            sequence = torch.cat(
                [
                    latent,
                    action,
                ],
                dim=1,
            )

        prediction = self.latent_predictor(sequence, sequence)

        return prediction[:, :-1]

    def forward(self, latent, actions):

        for action in actions.unbind(dim=1):
            latent = self.step(latent, action)

        return latent
