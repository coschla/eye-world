import gymnasium as gym
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from data.data_write import create_webdataset
from dataset.pre_process import ComposePreprocessor, Resize, Stack, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from Eval.gym_eval import GymManager
from models.action_net import ActionNet
from models.networks import ConvNet, UNet
from models.vjepa import (
    ActionEmbedding,
    Predictor,
    TransformerEncoder,
    TubeletEmbedding,
    VJEPAEncoder,
)
from trainers.action_predict import ActionTraining
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


with skip_run("run", "jepa_training") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    train_loader = dataloaders["train"]

    for x, y in train_loader:
        print("Train batch shape:", x.shape)  # [32, 4, 84, 84]
        break

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
    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)
    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)

    pred = Predictor(embed_dim, depth=4, heads=heads // 2, mlp_dim=mlp_dim)
    model = VJEPA(
        model=net,
        pred=pred,
        config=config,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader)


import random

from torch.utils.data import DataLoader, IterableDataset

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


with skip_run("flase", "jepa_trainers") as check, check():
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
import os

from gymnasium.wrappers import RecordVideo
from torch.utils.data import DataLoader, IterableDataset

with skip_run("skip", "action_condition_classifier") as check, check():
    game = config["games"][0]  # ONLY ONE GAME

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])

    logger = TensorBoardLogger("tb_logs", name=f"{game}/action_classifier/")

    # -----------------------------
    # Single dataset + dataloader
    # -----------------------------
    ds = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    train_loader = ds["train"]
    test_loader = ds["test"]
    loader = ds["train"]
    batch = next(iter(loader))

    _, _, actions = batch

    data_loaders = {"train": train_loader, "test": test_loader}

    # -----------------------------
    # Model
    # -----------------------------
    num_actions = config.get("num_actions", 18)
    net = ActionNet(num_actions)

    model = ActionTraining(hparams=config, net=net, data_loader=data_loaders)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="cpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model)

    class RuntimePreprocessor:
        def __init__(self, config):
            self.pipeline = ComposePreprocessor(
                [
                    Resize(config),
                    StackWithLabels(config),
                ]
            )

            # dummy placeholders (since env doesn't provide these)
            self.last_action = 0
            self.last_gaze = [(0, 0)]  # or whatever shape your code expects

        def step(self, obs):
            sample = (obs, self.last_gaze, self.last_action)

            stacked_img, stacked_gaze, stacked_action = self.pipeline(sample)

            # update last action (optional, but keeps things consistent)
            self.last_action = stacked_action[-1].item()

            return stacked_img

    # =========================
    # Load model
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_actions = 18  # MsPacman action space
    # model = ActionNet(num_actions)
    """
        checkpoint_path = "your_checkpoint.ckpt"  # <-- CHANGE THIS
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # If saved with PyTorch Lightning
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        """
    model.to(device)
    model.eval()

    # =========================
    # Setup env + video
    # =========================
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # record every episode
    )

    runtime_preprocessor = RuntimePreprocessor(config)

    obs, info = env.reset()
    state = runtime_preprocessor.step(obs)

    done = False
    total_reward = 0
    final_score = 0  # for Atari score if available

    # =========================
    # Run agent
    # =========================
    while not done:
        state_tensor = state.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = runtime_preprocessor.step(obs)
        total_reward += reward

        # Some Atari envs include score in info
        if "score" in info:
            final_score = info["score"]

    # =========================
    # Results
    # =========================
    print("Total reward:", total_reward)
    print("Final score:", final_score)

    env.close()


with skip_run("skip", "action_condition_classifier_multi_val") as check, check():
    datasets = []
    games = config["games"]

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])

    logger = TensorBoardLogger("tb_logs", name="multi_game/action_classifier/")
    data_loaders = {"train": [], "test": []}

    for game in games:
        ds = get_torch_dataloaders(game, config, preprocessor=preprocessor)

        data_loaders["train"].append(ds["train"])
        data_loaders["test"].append((game, ds["test"]))
        # data_loaders["test"].append(ds["test"])

    """
        games = config["games"]
        for game in games:
        # Logger (same as before, just different name)
        logger = TensorBoardLogger("tb_logs", name=f"{game}/action_classifier/")

        preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
        dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
        """

    class InterleavedDataset(IterableDataset):
        def __init__(self, datasets):
            self.datasets = datasets

        def __iter__(self):
            iterators = [iter(ds) for ds in self.datasets]

            while iterators:
                i = random.randint(0, len(iterators) - 1)
                try:
                    yield next(iterators[i])
                except StopIteration:
                    iterators.pop(i)

    # Extract datasets
    train_datasets = [dl.dataset for dl in data_loaders["train"]]

    print("===== ACTION SPACE DEBUG (WebDataset) =====")

    for i, ds in enumerate(train_datasets):
        actions_seen = set()

        for j, sample in enumerate(ds):
            _, _, actions = sample
            actions_seen.update(actions.flatten().tolist())

            if j >= 500:
                break

        print(f"Dataset {i} unique actions:", sorted(actions_seen))

    print("==========================================")

    print("=================================")

    # Combine them (interleaved instead of concatenated)
    combined_dataset = InterleavedDataset(train_datasets)

    # Rebuild a proper DataLoader
    example_loader = data_loaders["train"][0]

    data_loaders["train"] = DataLoader(
        combined_dataset,
        batch_size=example_loader.batch_size,
        shuffle=False,
        num_workers=example_loader.num_workers,
        pin_memory=getattr(example_loader, "pin_memory", False),
    )

    num_actions = config.get("num_actions", 18)
    num_actions = num_actions
    net = ActionNet(num_actions)

    # -----------------------------
    # Lightning module
    # -----------------------------
    model = ActionTraining(hparams=config, net=net, data_loader=data_loaders)

    # -----------------------------
    # Trainer (mirrors your setup)
    # -----------------------------
    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="cpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(model)

    class RuntimePreprocessor:
        def __init__(self, config):
            self.pipeline = ComposePreprocessor(
                [
                    Resize(config),
                    StackWithLabels(config),
                ]
            )

            # dummy placeholders (since env doesn't provide these)
            self.last_action = 0
            self.last_gaze = [(0, 0)]  # or whatever shape your code expects

        def step(self, obs):
            sample = (obs, self.last_gaze, self.last_action)

            stacked_img, stacked_gaze, stacked_action = self.pipeline(sample)

            # update last action (optional, but keeps things consistent)
            self.last_action = stacked_action[-1].item()

            return stacked_img

    # =========================
    # Load model
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_actions = 18  # MsPacman action space
    # model = ActionNet(num_actions)
    """
        checkpoint_path = "your_checkpoint.ckpt"  # <-- CHANGE THIS
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # If saved with PyTorch Lightning
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        """
    model.to(device)
    model.eval()

    # =========================
    # Setup env + video
    # =========================
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # record every episode
    )

    runtime_preprocessor = RuntimePreprocessor(config)

    obs, info = env.reset()
    state = runtime_preprocessor.step(obs)

    done = False
    total_reward = 0
    final_score = 0  # for Atari score if available

    # =========================
    # Run agent
    # =========================
    while not done:
        state_tensor = state.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = runtime_preprocessor.step(obs)
        total_reward += reward

        # Some Atari envs include score in info
        if "score" in info:
            final_score = info["score"]

    # =========================
    # Results
    # =========================
    print("Total reward:", total_reward)
    print("Final score:", final_score)

    env.close()

import torch.nn.functional as F

with skip_run("skip", "jepa_rollout_trainer") as check, check():
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

    rollout_steps = config.get("rollout_steps", 4)

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

            # -----------------------------
            # FREEZE EVERYTHING EXCEPT PREDICTOR
            # -----------------------------
            self.model.tubelet_embed.requires_grad_(False)
            self.model.student.requires_grad_(False)
            self.action_embed.requires_grad_(False)

        # -----------------------------
        # one-step transition
        # -----------------------------
        """
        def predict_next(self, latent, action):
            with torch.no_grad():
                a = self.action_embed(action).unsqueeze(1)
                seq = torch.cat([latent, a], dim=1)

            out = self.latent_predictor(seq, seq)
            return out[:, :-1, :]
            """

        def predict_next(self, latent, action):
            with torch.no_grad():
                # action: [B, ...] -> must become [B, D]
                a = self.action_embed(action)

                # collapse spatial dims if they exist
                if a.ndim > 2:
                    a = a.mean(dim=tuple(range(1, a.ndim - 1)))  # [B, D]

                # make it a token
                a = a.unsqueeze(1)  # [B, 1, D]

                seq = torch.cat([latent, a], dim=1)

            out = self.latent_predictor(seq, seq)
            return out[:, :-1, :]

        # -----------------------------
        # training
        # -----------------------------
        def training_step(self, batch, batch_idx):
            img, actions, *_ = batch
            context_frames = config["context_frames"]

            # -----------------------------
            # Encode initial context (student)
            # -----------------------------
            with torch.no_grad():
                context = img[:, :context_frames].unsqueeze(2)
                z = self.model.tubelet_embed(context)
                z = self.model.student(z)

            # -----------------------------
            # Teacher rollout targets
            # -----------------------------
            future = img[:, context_frames : context_frames + self.rollout_steps]

            targets = []

            with torch.no_grad():
                context_window = img[:, :context_frames]

                for t in range(self.rollout_steps):
                    # Slide window forward by one frame
                    next_frame = future[:, t].unsqueeze(1)

                    context_window = torch.cat(
                        [context_window[:, 1:], next_frame],
                        dim=1,
                    )

                    # Encode the new 4-frame window
                    z_t = self.model.tubelet_embed(context_window.unsqueeze(2))
                    z_t = self.model.student(z_t)

                    targets.append(z_t)

            # -----------------------------
            # Rollout with trainable predictor
            # -----------------------------
            preds = []
            latent = z

            for t in range(self.rollout_steps):
                a_t = actions[:, context_frames + t]
                latent = self.predict_next(latent, a_t)
                preds.append(latent)

            # -----------------------------
            # LOSS
            # -----------------------------
            loss = sum(F.smooth_l1_loss(p, g) for p, g in zip(preds, targets))
            loss = loss / self.rollout_steps

            self.log("loss/rollout_dynamics", loss, on_epoch=True, prog_bar=True)

            return loss

        # -----------------------------
        # ONLY predictor is optimized
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
        "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_4/checkpoints/epoch=49-step=164350.ckpt",
        map_location="cpu",
    )

    state_dict = ckpt["state_dict"]

    # Remove obsolete teacher weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, dataloaders["train"])


with skip_run("skip", "jepa_rollout_trainer_with_validation") as check, check():
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

    rollout_steps = config.get("rollout_steps", 4)

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
        "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_4/checkpoints/epoch=49-step=164350.ckpt",
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


import gymnasium as gym
import torch
import torch.nn as nn

# ============================================================
# Atari Environment
# ============================================================


class GymManager:
    """
    Executes actions in the real Atari environment.

    Used ONLY for generating the training target
    (future score - current score).

    The model never receives observations from here after the
    initial latent has been created.
    """

    def __init__(
        self,
        config,
        env_name="ALE/MsPacman-v5",
    ):
        self.env = gym.make(env_name)
        self.preprocessor = ComposePreprocessor(config)

    def reset(self):

        obs, _ = self.env.reset()

        return self.preprocessor.reset(obs)

    def step(self, action):

        obs, reward, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated

        state = self.preprocessor.step(obs)

        return state, reward, done


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


# ============================================================
# Score Difference Predictor
# ============================================================


class ScoreDeltaPredictor(nn.Module):
    """
    Predicts

        future_score - current_score

    Inputs
    ------
    current latent
    predicted future latent
    current score

    Outputs
    -------
    predicted score delta
    """

    def __init__(
        self,
        embed_dim=768,
        heads=12,
        depth=4,
        mlp_dim=3072,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
        )

        self.current_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.future_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.score_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        current_latent,
        future_latent,
        current_score,
    ):

        current = self.current_encoder(current_latent)
        future = self.future_encoder(future_latent)

        # Pool over however many tubelets exist.
        # Works with ANY number of tokens.
        current = current.mean(dim=1)
        future = future.mean(dim=1)

        score = self.score_embedding(current_score.unsqueeze(-1).float())

        features = torch.cat(
            [
                current,
                future,
                score,
            ],
            dim=-1,
        )

        delta = self.head(features)

        return delta.squeeze(-1)


# ============================================================
# RL Training Wrapper
# ============================================================


class ScoreDeltaTrainer(nn.Module):
    """
    Uses

        JEPA Encoder
        +
        Latent Predictor
        +
        Atari Environment

    to train the score difference predictor.
    """

    def __init__(
        self,
        jepa_model,
        action_embed,
        latent_predictor,
        config,
    ):
        super().__init__()

        self.model = jepa_model
        self.rollout = LatentRollout(
            latent_predictor,
            action_embed,
        )

        self.score_predictor = ScoreDeltaPredictor()

        self.config = config

    def encode(self, stacked_frames):
        """
        Encode only the CURRENT observation.

        Shape:
            B x context_frames x C x H x W
        """

        with torch.no_grad():
            latent = self.model.tubelet_embed(stacked_frames.unsqueeze(2))

            latent = self.model.student(latent)

        return latent

    def training_step(
        self,
        current_frames,
        current_score,
        actions,
        gym_manager,
    ):
        """
        current_frames : initial stack
        current_score  : score before rollout
        actions        : actions for cycle_steps
        """

        device = current_frames.device

        # -------------------------------------
        # Encode current state
        # -------------------------------------

        current_latent = self.encode(current_frames)

        # -------------------------------------
        # Predict future latent
        # -------------------------------------

        future_latent = self.rollout(
            current_latent,
            actions[:, : self.config["cycle_steps"]],
        )

        # -------------------------------------
        # Predict score difference
        # -------------------------------------

        predicted_delta = self.score_predictor(
            current_latent,
            future_latent,
            current_score,
        )

        # -------------------------------------
        # Real environment rollout
        # -------------------------------------

        target_delta = []

        batch_size = actions.shape[0]

        for b in range(batch_size):
            score = current_score[b].item()

            for action in actions[b, : self.config["cycle_steps"]]:
                _, reward, done = gym_manager.step(int(action))

                score += reward

                if done:
                    break

            target_delta.append(score - current_score[b].item())

        target_delta = torch.tensor(
            target_delta,
            device=device,
            dtype=torch.float32,
        )

        # -------------------------------------
        # Loss
        # -------------------------------------

        loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
        )

        return loss

    # ============================================================
    # Example Usage
    # ============================================================


gym_manager = GymManager(config)

trainer = ScoreDeltaTrainer(
    jepa_model=base_model.model,
    action_embed=base_model.action_embed,
    latent_predictor=base_model.latent_predictor,
    config=config,
)

loss = trainer.training_step(
    current_frames,
    current_score,
    actions,
    gym_manager,
)
