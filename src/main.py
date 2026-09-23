import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from data.data_write import create_webdataset
from dataset.pre_process import ComposePreprocessor, Resize, Stack, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from evaluate.gym_eval import GymManager
from models.action_net import ActionNet
from models.networks import ConvNet, UNet
from models.vjepa import (
    ActionEmbedding,
    Predictor,
    TransformerEncoder,
    TubeletEmbedding,
    VJEPAEncoder,
)
from trainers.action_classifier import ActionTraining
from trainers.gaze_predict import GazeTraining
from trainers.jepa import VJEPA, ActionConditionVJEPA
from trainers.jepa_rollout import RolloutActionJEPA
from utils import InterleavedDataset, skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

CHECKPOINT_DIR = Path(config["action_classifier_checkpoint_dir"])


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


with skip_run("skip", "jepa_rollout_with_validation") as check, check():
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


with skip_run("skip", "train_action_classifier") as check, check():
    game = config["games"]
    games_name = "_".join(game)
    training_preprocessor = ComposePreprocessor(
        [
            Resize(config),
            StackWithLabels(config),
        ]
    )

    dataset_loaders = get_torch_dataloaders(
        game,
        config,
        preprocessor=training_preprocessor,
    )

    if "train" not in dataset_loaders:
        raise KeyError("get_torch_dataloaders() did not return a 'train' loader.")

    data_loaders = {
        "train": dataset_loaders["train"],
    }

    if "val" in dataset_loaders:
        data_loaders["val"] = dataset_loaders["val"]

    if "test" in dataset_loaders:
        data_loaders["test"] = dataset_loaders["test"]

    num_actions = int(config["num_actions"])

    """if num_actions != 9:
        raise ValueError(
            "This training setup expects num_actions: 9, "
            f"but config contains {num_actions}."
        )"""

    ##############################################################################
    from collections import Counter

    from trainers.utils import atari_to_gym

    raw_counts = Counter()
    mapped_counts = Counter()

    for batch in data_loaders["train"]:
        _, _, actions = batch

        actions = torch.as_tensor(actions)

        if actions.ndim >= 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        if actions.ndim > 1:
            actions = actions[:, -1]

        raw_counts.update(actions.cpu().tolist())

        mapped = atari_to_gym(actions)
        mapped_counts.update(mapped.cpu().tolist())

    print("\nRAW ALE ACTION DISTRIBUTION")
    raw_names = {
        0: "NOOP",
        1: "FIRE",
        3: "RIGHT",
        4: "LEFT",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
    }

    total = sum(raw_counts.values())

    for action, count in sorted(raw_counts.items()):
        print(
            action,
            raw_names.get(action, "UNKNOWN"),
            count,
            f"{100 * count / total:.2f}%",
        )

    print("\nGYM ACTION DISTRIBUTION")

    gym_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE",
    }

    total = sum(mapped_counts.values())

    for action in range(6):
        count = mapped_counts[action]
        print(action, gym_names[action], count, f"{100 * count / total:.2f}%")
    ##############################################################################

    action_network = ActionNet(
        num_actions=num_actions,
    )

    training_model = ActionTraining(
        hparams=config,
        net=action_network,
        data_loader=data_loaders,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"{games_name}-action-classifier-best-curent-test",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"{games_name}/action_classifier",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config["epochs"]),
        log_every_n_steps=10,
    )

    trainer.fit(training_model)

    if data_loaders.get("test") is not None:
        trainer.test(
            training_model,
            dataloaders=data_loaders["test"],
        )

    print("\nTraining complete")
    print(
        "Last checkpoint:",
        checkpoint_callback.last_model_path,
    )

from collections import Counter

import gymnasium as gym

with skip_run("run", "action_classifier_in_gym_recording") as check, check():
    runtime_config = dict(config)

    runtime_config["action_classifier_checkpoint"] = (
        "checkpoints/action_classifier/space_invaders-action-classifier-best-curent-test.ckpt"
    )

    num_episodes = int(runtime_config.get("gym_eval_episodes", 10))

    max_steps = int(runtime_config.get("gym_max_steps", 100_00))

    preprocessor = ComposePreprocessor(
        [
            Resize(config),
            StackWithLabels(config),
        ]
    )

    action_net = ActionNet(num_actions=int(config["num_actions"]))

    manager = GymManager(
        config=runtime_config,
        preprocessor_pipeline=preprocessor,
        action_net=action_net,
        env_name="ALE/SpaceInvaders-v5",
        record_video=True,
        video_folder="./video/action_classifier",
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    # ---------------------------------------------------------
    # SPACE INVADERS ACTION NAMES
    # ---------------------------------------------------------

    gym_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE",
    }

    num_actions = int(config["num_actions"])

    # ---------------------------------------------------------
    # WRAPPER THAT CAPTURES THE REAL ACTION SENT TO GYM
    # ---------------------------------------------------------

    class ActionTrackingWrapper(gym.Wrapper):
        def step(self, action):
            # Convert torch/numpy scalar if necessary.
            if hasattr(action, "item"):
                action = action.item()

            action = int(action)

            # This is the ACTUAL action being sent to Atari.
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Add it to info so the evaluation loop can track it.
            info = dict(info)
            info["action"] = action

            return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------
    # FIND GymManager's ENVIRONMENT AND WRAP IT
    # ---------------------------------------------------------

    if hasattr(manager, "env"):
        manager.env = ActionTrackingWrapper(manager.env)

    elif hasattr(manager, "_env"):
        manager._env = ActionTrackingWrapper(manager._env)

    else:
        raise RuntimeError(
            "Could not find GymManager environment. "
            "Expected manager.env or manager._env."
        )

    # ---------------------------------------------------------
    # TRACKING STORAGE
    # ---------------------------------------------------------

    episode_rewards = []

    total_action_counts = Counter()
    episode_action_counts = []

    print("\n" + "=" * 60)
    print("SPACE INVADERS ACTION MAPPING")
    print("=" * 60)

    for action_id in range(num_actions):
        print(f"Action {action_id}: {gym_names.get(action_id, f'ACTION_{action_id}')}")

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    try:
        for episode in range(num_episodes):
            state = manager.reset()

            print("\n" + "=" * 60)
            print(f"Episode {episode + 1}/{num_episodes} started")
            print("=" * 60)

            print(
                "Initial state shape:",
                tuple(state.shape),
            )

            total_reward = 0.0
            step_count = 0
            final_info = {}
            done = False

            action_counts = Counter()

            while not done and step_count < max_steps:
                state, reward, done, info = manager.step()

                # -------------------------------------------------
                # GET ACTUAL ACTION SENT TO GYM
                # -------------------------------------------------

                if "action" not in info:
                    raise RuntimeError(
                        "ActionTrackingWrapper did not receive an "
                        "action from the environment.\n"
                        f"Available info keys: {list(info.keys())}"
                    )

                action = int(info["action"])

                # -------------------------------------------------
                # VERIFY VALID ACTION
                # -------------------------------------------------

                if action < 0 or action >= num_actions:
                    raise RuntimeError(
                        f"Invalid action: {action}. Expected 0-{num_actions - 1}."
                    )

                # -------------------------------------------------
                # COUNT ACTION
                # -------------------------------------------------

                action_counts[action] += 1
                total_action_counts[action] += 1

                # -------------------------------------------------
                # DEBUG FIRST 30 ACTIONS
                # -------------------------------------------------

                if episode == 0 and step_count < 30:
                    action_name = gym_names.get(
                        action,
                        f"ACTION_{action}",
                    )

                    print(
                        f"Step {step_count:4d} | "
                        f"Action {action} "
                        f"({action_name:10s}) | "
                        f"Reward {float(reward):7.2f}"
                    )

                # -------------------------------------------------
                # METRICS
                # -------------------------------------------------

                total_reward += float(reward)
                step_count += 1

                final_info = info

            # -----------------------------------------------------
            # SAVE EPISODE
            # -----------------------------------------------------

            episode_rewards.append(total_reward)

            episode_action_counts.append(action_counts.copy())

            print()
            print(f"Episode {episode + 1} finished")

            print(
                "Steps:",
                step_count,
            )

            print(
                "Total reward:",
                total_reward,
            )

            if "score" in final_info:
                print(
                    "Final score:",
                    final_info["score"],
                )

            # -----------------------------------------------------
            # ACTION USAGE THIS EPISODE
            # -----------------------------------------------------

            print("\nAction usage:")

            for action_id in range(num_actions):
                count = action_counts[action_id]

                percentage = 100.0 * count / step_count if step_count > 0 else 0.0

                action_name = gym_names.get(
                    action_id,
                    f"ACTION_{action_id}",
                )

                print(
                    f"  Action {action_id} "
                    f"({action_name:10s}): "
                    f"{count:6d} times "
                    f"({percentage:6.2f}%)"
                )

            if step_count >= max_steps and not done:
                print(f"Episode stopped because it reached gym_max_steps={max_steps}.")

        # ---------------------------------------------------------
        # REWARD SUMMARY
        # ---------------------------------------------------------

        mean_reward = statistics.mean(episode_rewards)

        reward_std = statistics.pstdev(episode_rewards)

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"Evaluated {num_episodes} episodes")

        print(
            "Per-episode rewards:",
            episode_rewards,
        )

        print(f"Mean reward: {mean_reward:.2f} (std: {reward_std:.2f})")

        # ---------------------------------------------------------
        # TOTAL ACTION USAGE
        # ---------------------------------------------------------

        total_actions = sum(total_action_counts.values())

        print("\n" + "=" * 60)
        print("TOTAL ACTION USAGE")
        print("=" * 60)

        print(f"Total actions executed: {total_actions}")

        for action_id in range(num_actions):
            count = total_action_counts[action_id]

            percentage = 100.0 * count / total_actions if total_actions > 0 else 0.0

            action_name = gym_names.get(
                action_id,
                f"ACTION_{action_id}",
            )

            print(
                f"Action {action_id} "
                f"({action_name:10s}): "
                f"{count:6d} times "
                f"({percentage:6.2f}%)"
            )

        # ---------------------------------------------------------
        # PER-EPISODE SUMMARY
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("ACTION COUNTS BY GAMEPLAY SESSION")
        print("=" * 60)

        for episode_id, counts in enumerate(
            episode_action_counts,
            start=1,
        ):
            print(f"\nEpisode {episode_id}")

            episode_total = sum(counts.values())

            for action_id in range(num_actions):
                count = counts[action_id]

                percentage = 100.0 * count / episode_total if episode_total > 0 else 0.0

                action_name = gym_names.get(
                    action_id,
                    f"ACTION_{action_id}",
                )

                print(
                    f"  Action {action_id} "
                    f"({action_name:10s}): "
                    f"{count:6d} "
                    f"({percentage:6.2f}%)"
                )

        # ---------------------------------------------------------
        # EXISTING ACTION NET REPORT
        # ---------------------------------------------------------

        if hasattr(
            manager.action_net,
            "report",
        ):
            manager.action_net.report()

    finally:
        manager.close()
