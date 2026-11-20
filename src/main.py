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
from dataset.utils import (
    compute_action_class_weights,
    get_train_test_files,
    get_train_val_test_files,
)
from evaluate.encoder_check import check_encoder
from evaluate.gym_eval import GymManager, evaluate_policy
from evaluate.utils import ALE_ACTION_NAMES
from models.action_jepa_history import ActionNet as JEPAActionNet
from models.action_net import ACPredictor, ActionNet
from models.networks import ConvNet, UNet
from models.jepa import (
    Predictor,
    TransformerEncoder,
    TubeletEmbedding,
    JEPAEncoder,
)
from trainers.gaze_predict import GazeTraining
from trainers.jepa import JEPA
from trainers.action_classifier import ActionHistoryTraining
from trainers.jepa_rollout import RolloutActionJEPA, load_encoder_weights
from utils import InterleavedDataset, skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

LOG_DIR = config["log_dir"]
VIDEO_DIR = Path(config["video_dir"])


with skip_run("skip", "shard_creation") as check, check():
    for game in config["games"]:
        create_webdataset(game, config)


with skip_run("skip", "torch_dataset_verify") as check, check():
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
    logger = TensorBoardLogger(LOG_DIR, name=f"{game}/gaze_prediction/")
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
    logger = TensorBoardLogger(LOG_DIR, name=f"{game}/gaze_prediction/")
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


with skip_run("skip", "train_cnn_classifier") as check, check():
    cnn_config = {**config, "stack_length": 3, "epochs": 25}  # single frame, as in the Atari-HEAD paper
    game = cnn_config["games"]
    games_name = "_".join(game)
    num_actions = int(cnn_config["num_actions"])
    preprocessor = ComposePreprocessor([Resize(cnn_config), StackWithLabels(cnn_config)])
    data_loaders = get_torch_dataloaders(game, cnn_config, preprocessor, use_val=True)

    train_files = [f for g in game for f in get_train_val_test_files(g, cnn_config)[0]]
    class_weights = compute_action_class_weights(
        train_files,
        num_actions,
        cnn_config.get("class_weight_max_ratio"),
        int(cnn_config["class_weight_min_count"]),
    )
    in_channels = int(cnn_config["stack_length"]) * (1 if cnn_config.get("grey_scale", True) else 3)
    training_model = ActionHistoryTraining(
        cnn_config, ActionNet(num_actions, in_channels), data_loaders, class_weights
    )

    checkpoint_callback = ModelCheckpoint(
        filename=f"{games_name}-cnn-best", monitor="val_balanced_accuracy", mode="max", save_top_k=1
    )
    gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        logger=TensorBoardLogger(LOG_DIR, name=f"{games_name}/cnn_classifier"),
        callbacks=[checkpoint_callback],
        accelerator="gpu" if gpu else "cpu",
        devices=2,
        max_epochs=int(cnn_config["epochs"]),
        log_every_n_steps=10,
        gradient_clip_val=float(cnn_config["grad_clip"]),
    )
    trainer.fit(training_model)
    trainer.test(training_model, dataloaders=data_loaders["test"], ckpt_path="best")
    print("Best checkpoint:", checkpoint_callback.best_model_path)


with skip_run("run", "evaluate_cnn_classifier") as check, check():
    # Same single-frame setup as train_cnn_classifier. Freeway has no FIRE, so no auto-FIRE.
    cnn_config = {**config, "stack_length": 1, "gym_auto_fire": True, "gym_repeat_action_probability": 0.0}
    cnn_config["action_classifier_checkpoint"] = "tb_logs/freeway/cnn_classifier/version_4/checkpoints/freeway-cnn-best.ckpt"

    manager = GymManager(
        config=cnn_config,
        preprocessor_pipeline=ComposePreprocessor(
            [Resize(cnn_config), StackWithLabels(cnn_config)]
        ),
        action_net=ActionNet(int(cnn_config["num_actions"]), in_channels=cnn_config['stack_length']),
        record_video=bool(cnn_config["gym_record_video"]),
        video_folder=str(VIDEO_DIR / "cnn_classifier"),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    evaluate_policy(
        manager,
        num_episodes=int(cnn_config["gym_eval_episodes"]),
        max_steps=int(cnn_config["gym_max_steps"]),
    )


with skip_run("skip", "jepa_training_multi_game") as check, check():
    logger = TensorBoardLogger(
        LOG_DIR,
        name="multi_game/vjepa_world_model/",
    )

    preprocessor = ComposePreprocessor(
        [
            Resize(config),
            Stack(config),
        ]
    )

    # Load all game datasets
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

    # Build train dataloader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=example_loader.batch_size,
        shuffle=False,
        num_workers=example_loader.num_workers,
        pin_memory=getattr(example_loader, "pin_memory", False),
    )

    # Verify batch shape
    for x, y in train_loader:
        print("Train batch shape:", x.shape)
        print("Label shape:", y.shape)
        break

    # Model configuration
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

    net = JEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    pred = Predictor(
        embed_dim,
        depth=4,
        heads=heads // 2,
        mlp_dim=mlp_dim,
    )

    model = JEPA(
        model=net,
        pred=pred,
        config=config,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader)


with skip_run("skip", "encoder_check") as check, check():
    # Do the frozen encoder's latents vary with the frame? (see evaluate/encoder_check.py)
    ac_config = {**config, "stack_length": int(config["context_frames"]) + 4}
    preprocessor = ComposePreprocessor([Resize(ac_config), StackWithLabels(ac_config)])
    dataloaders = get_torch_dataloaders(config["games"][0], ac_config, preprocessor)

    tubelet_embed = TubeletEmbedding(config, 1 if config.get("grey_scale", True) else 3, 768, config.get("size_x", 84))
    net = JEPAEncoder(tubelet_embed, TransformerEncoder(768, depth=12, heads=12, mlp_dim=3072))

    config["jepa_checkpoint"] = 'tb_logs/multi_game/vjepa_world_model/version_0/checkpoints/epoch=14-step=21645.ckpt'
    load_encoder_weights(net, config["jepa_checkpoint"])
    check_encoder(net, dataloaders["test"], ac_config)


with skip_run("skip", "train_action_classifier") as check, check():
    game = config["games"]
    config['epochs'] = 20
    games_name = "_".join(game)
    num_actions = int(config["num_actions"])
    assert num_actions == len(ALE_ACTION_NAMES), "classifier uses the full ALE action space"

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
    data_loaders = get_torch_dataloaders(game, config, preprocessor, use_val=True)

    train_files = [f for g in game for f in get_train_val_test_files(g, config)[0]]
    class_weights = compute_action_class_weights(
        train_files,
        num_actions,
        config.get("class_weight_max_ratio"),
        int(config["class_weight_min_count"]),
    )
    for a, w in enumerate(class_weights.tolist()):
        print(a, ALE_ACTION_NAMES[a], f"{w:.3f}")

    freeze = bool(config["freeze_encoder"])
    config["jepa_checkpoint"] = 'tb_logs/multi_game/vjepa_world_model/version_0/checkpoints/epoch=14-step=21645.ckpt'
    action_network = JEPAActionNet(
        num_actions=num_actions,
        config=config,
        jepa_checkpoint=config["jepa_checkpoint"],
        freeze_tubelet=freeze,
        freeze_student=freeze,
    )
    training_model = ActionHistoryTraining(config, action_network, data_loaders, class_weights)

    checkpoint_callback = ModelCheckpoint(
        filename=f"{games_name}-action-jepa-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        logger=TensorBoardLogger(LOG_DIR, name=f"{games_name}/action_jepa_classifier"),
        callbacks=[checkpoint_callback],
        accelerator="gpu" if gpu else "cpu",
        devices=1,
        max_epochs=int(config["epochs"]),
        precision="bf16-mixed" if gpu else 32,
        log_every_n_steps=10,
        gradient_clip_val=float(config["grad_clip"]),
    )
    trainer.fit(training_model)
    trainer.test(training_model, dataloaders=data_loaders["test"], ckpt_path="best")
    print("Best checkpoint:", checkpoint_callback.best_model_path)


with skip_run("skip", "evaluate_action_model") as check, check():
    config["action_classifier_checkpoint"] = "tb_logs/breakout/action_jepa_classifier/version_2/checkpoints/breakout-action-jepa-best.ckpt"
    runtime_config = dict(config)

    manager = GymManager(
        config=runtime_config,
        preprocessor_pipeline=ComposePreprocessor(
            [
                Resize(runtime_config),
                StackWithLabels(runtime_config),
            ]
        ),
        action_net=JEPAActionNet(
            num_actions=int(runtime_config["num_actions"]),
            config=runtime_config,
            # Encoder weights come from the classifier checkpoint that
            # RuntimeActionNet loads (strict) from action_classifier_checkpoint.
            jepa_checkpoint=None,
            freeze_tubelet=True,
            freeze_student=True,
        ),
        record_video=bool(runtime_config["gym_record_video"]),
        video_folder=str(VIDEO_DIR / "action_jepa_classifier"),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    evaluate_policy(
        manager,
        num_episodes=int(runtime_config.get("gym_eval_episodes", 10)),
        max_steps=int(runtime_config.get("gym_max_steps", 10_000)),
    )


with skip_run("skip", "jepa_rollout_with_validation") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger(LOG_DIR, name=f"{game}/vjepa_rollout_world_model/")

    # Each sample holds the context window plus ac_num_steps future frames.
    # stack_length stays 4 in the shared config because pretraining needs it.
    ac_config = {
        **config,
        "stack_length": int(config["context_frames"]) + int(config["ac_num_steps"]),
    }

    preprocessor = ComposePreprocessor([Resize(ac_config), StackWithLabels(ac_config)])
    dataloaders = get_torch_dataloaders(game, ac_config, preprocessor=preprocessor)

    # MODEL: frozen pretrained encoder + action-conditioned predictor
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

    net = JEPAEncoder(tubelet_embed=tubelet_embed, student=student)

    # Pretrained JEPA tubelet embedding + student (raises if the file is missing).
    load_encoder_weights(net, config["jepa_checkpoint"])

    predictor = ACPredictor(
        embed_dim=embed_dim,
        dim=int(config["ac_dim"]),
        depth=int(config["ac_depth"]),
        heads=int(config["ac_heads"]),
        num_actions=int(config["num_actions"]),
        num_patches=tubelet_embed.num_patches,
        max_steps=int(config["ac_num_steps"]),
    )

    model = RolloutActionJEPA(net, predictor, ac_config)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ModelCheckpoint(monitor="val/ar_loss", mode="min", save_top_k=1)],
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


with skip_run("skip", "action_classifier_in_gym_recording") as check, check():
    runtime_config = dict(config)

    runtime_config["action_classifier_checkpoint"] = config["action_classifier_cnn_checkpoint"]

    manager = GymManager(
        config=runtime_config,
        preprocessor_pipeline=ComposePreprocessor(
            [
                Resize(config),
                StackWithLabels(config),
            ]
        ),
        action_net=ActionNet(num_actions=int(config["num_actions"])),
        record_video=bool(runtime_config["gym_record_video"]),
        video_folder=str(VIDEO_DIR / "action_classifier"),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    evaluate_policy(
        manager,
        num_episodes=int(runtime_config.get("gym_eval_episodes", 10)),
        max_steps=int(runtime_config.get("gym_max_steps", 10_000)),
    )
