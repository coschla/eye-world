import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from src.dataset.pre_process import ComposePreprocessor, Resize, StackWithLabels
from src.dataset.torch_dataset import get_torch_dataloaders
from src.Eval.gym_eval import GymManager, RecordingGymManager, RuntimePreprocessor
from src.models.utils import atari_to_gym
from src.utils import skip_run

# from src.utils skip_run

# ================================================================
# Configuration
# ================================================================

CONFIG_PATH = Path("configs/config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

game = config["games"][0]

CHECKPOINT_DIR = Path(
    config.get(
        "action_classifier_checkpoint_dir",
        "checkpoints/action_classifier",
    )
)

CHECKPOINT_PATH = Path(
    config.get(
        "action_classifier_checkpoint",
        CHECKPOINT_DIR / "last.ckpt",
    )
)


# ================================================================
# Action network
# ================================================================


class ActionNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=8,
                stride=4,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # An 84x84 input produces 64 x 7 x 7 = 3136 features.
        self.num_actions = num_actions
        self.num_predictions = 4

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                self.num_predictions * self.num_actions,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        # [batch, 36] -> [batch, 4, 9]
        x = x.reshape(
            x.shape[0],
            self.num_predictions,
            self.num_actions,
        )

        return x


# ================================================================
# Lightning training module
# ================================================================


class ActionTraining(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        net: nn.Module,
        data_loader: dict,
    ):
        super().__init__()

        self.net = net
        self.data_loaders = data_loader

        self.learning_rate = float(hparams.get("learning_rate", 1e-3))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    @staticmethod
    def _prepare_images(images: torch.Tensor) -> torch.Tensor:
        """
        Accept either:

            [batch, 4, 84, 84]

        or:

            [batch, 4, 1, 84, 84]

        ActionNet requires [batch, 4, 84, 84].
        """
        images = images.float()

        if images.ndim == 5 and images.shape[2] == 1:
            images = images.squeeze(2)

        if images.ndim != 4:
            raise ValueError(
                "Training images must have shape "
                "[batch, frames, height, width], but received "
                f"{tuple(images.shape)}."
            )

        if images.shape[1] != 4:
            raise ValueError(
                "ActionNet requires four stacked frames, but received "
                f"shape {tuple(images.shape)}."
            )

        return images

    @staticmethod
    def _prepare_actions(actions: torch.Tensor) -> torch.Tensor:
        """
        Prepare four action targets per sample.

        Input:
            [batch, 4]
            [batch, 4, 1]

        Output:
            [batch, 4] containing Gym action IDs 0-8.
        """
        actions = torch.as_tensor(actions)

        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        if actions.ndim != 2:
            raise ValueError(
                "Expected actions with shape [batch, 4], "
                f"but received {tuple(actions.shape)}."
            )

        if actions.shape[1] != 4:
            raise ValueError(
                "Expected four actions per sample, "
                f"but received shape {tuple(actions.shape)}."
            )

        actions = atari_to_gym(actions)

        return actions.long()

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        images, _, actions = batch

        images = self._prepare_images(images)
        targets = self._prepare_actions(actions)

        logits = self.net(images)

        # Expected:
        # logits:  [batch, 4, 9]
        # targets: [batch, 4]
        if logits.ndim != 3:
            raise ValueError(
                "Expected logits with shape [batch, 4, 9], "
                f"but received {tuple(logits.shape)}."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}."
            )

        if logits.shape[2] != 9:
            raise ValueError(
                f"Expected nine Gym action classes, but received {logits.shape[2]}."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, 9),
            targets.reshape(-1),
        )

        predicted_actions = logits.argmax(dim=2)

        accuracy = (predicted_actions == targets).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        return loss

    """
    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        images, _, actions = batch

        images = self._prepare_images(images)
        targets = self._prepare_actions(actions)

        logits = self.net(images)

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "Batch size mismatch between predictions and targets: "
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}."
            )

        if logits.shape[1] != 9:
            raise ValueError(
                "The network must produce nine Gym action logits, "
                f"but produced shape {tuple(logits.shape)}."
            )

        loss = F.cross_entropy(logits, targets)

        predicted_actions = logits.argmax(dim=1)
        accuracy = (predicted_actions == targets).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )

        return loss"""

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return self.data_loaders["train"]

    def val_dataloader(self):
        loader = self.data_loaders.get("val")

        if loader is None:
            return []

        return loader

    def test_dataloader(self):
        loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader


# ================================================================
# GymManager-compatible checkpoint network
# ================================================================


class CheckpointActionNet:
    """
    Load a trained ActionNet and provide GymManager's required:

        act(data_packet) -> int

    The network has already been trained on Gym action IDs 0-8.
    No Atari-to-Gym conversion occurs here.
    """

    def __init__(
        self,
        config: dict,
        action_space: Any,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(config["action_classifier_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Action-classifier checkpoint was not found: {checkpoint_path}"
            )

        num_actions = int(config["num_actions"])

        if num_actions != action_space.n:
            raise ValueError(
                "The trained model and Gym environment have different "
                "action-space sizes. "
                f"config num_actions={num_actions}, "
                f"Gym action_space.n={action_space.n}."
            )

        self.model = ActionNet(num_actions=num_actions)

        checkpoint = self._load_checkpoint(checkpoint_path)
        model_state = self._extract_action_net_state(checkpoint)

        self.model.load_state_dict(
            model_state,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Inference device: {self.device}")
        print(f"Gym action count: {action_space.n}")

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> dict:
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            return torch.load(
                checkpoint_path,
                map_location=self.device,
            )

    def _extract_action_net_state(
        self,
        checkpoint: dict,
    ) -> dict:
        """
        Extract ActionNet parameters from a Lightning checkpoint.

        ActionTraining stores the network as self.net, so its checkpoint
        normally contains keys such as:

            net.conv_layers.0.weight
            net.fc_layers.2.bias
        """
        checkpoint_state = checkpoint.get(
            "state_dict",
            checkpoint,
        )

        expected_keys = set(self.model.state_dict().keys())

        extracted_state = {}

        possible_prefixes = (
            "",
            "net.",
            "model.",
            "model.net.",
            "module.",
            "module.net.",
        )

        for original_key, value in checkpoint_state.items():
            for prefix in possible_prefixes:
                if prefix and original_key.startswith(prefix):
                    candidate = original_key[len(prefix) :]
                else:
                    candidate = original_key

                if candidate in expected_keys:
                    extracted_state[candidate] = value
                    break

        missing_keys = expected_keys - set(extracted_state.keys())

        unexpected_shapes = []

        expected_state = self.model.state_dict()

        for key, value in extracted_state.items():
            if value.shape != expected_state[key].shape:
                unexpected_shapes.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(expected_state[key].shape),
                    )
                )

        if missing_keys:
            missing_text = "\n".join(f"  - {key}" for key in sorted(missing_keys))

            raise RuntimeError(
                f"The checkpoint is missing ActionNet parameters:\n{missing_text}"
            )

        if unexpected_shapes:
            shape_text = "\n".join(
                f"  - {key}: checkpoint={found}, expected={expected}"
                for key, found, expected in unexpected_shapes
            )

            raise RuntimeError(
                "The checkpoint architecture does not match ActionNet.\n"
                "This commonly occurs when loading an old 18-action "
                "checkpoint into the new nine-action model.\n"
                f"{shape_text}"
            )

        return extracted_state

    def act(self, data_packet: Any) -> int:
        state = torch.as_tensor(
            data_packet,
            dtype=torch.float32,
            device=self.device,
        )

        # RuntimePreprocessor returns [4, 1, 84, 84].
        # ActionNet requires [batch, 4, 84, 84].
        if state.ndim == 4 and state.shape[0] == 4 and state.shape[1] == 1:
            state = state.squeeze(1)

        if state.ndim == 3:
            state = state.unsqueeze(0)

        if state.ndim != 4:
            raise ValueError(
                "ActionNet expected a runtime input with shape "
                "[batch, 4, height, width], but received "
                f"{tuple(state.shape)}."
            )

        if state.shape[1] != 4:
            raise ValueError(
                "ActionNet expected four stacked frames, but received "
                f"shape {tuple(state.shape)}."
            )

        with torch.inference_mode():
            logits = self.model(state)

            # Four action predictions: [1, 4, 9]
            predicted_actions = logits.argmax(dim=2)

            # Use the prediction associated with the newest frame.
            action = predicted_actions[0, -1]

        return int(action.item())


# ================================================================
# Skip run 1: train and save the network
# ================================================================

with (
    skip_run(
        "skip",
        "train_action_classifier",
    ) as check,
    check(),
):
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

    if num_actions != 9:
        raise ValueError(
            "This training setup expects num_actions: 9, "
            f"but config contains {num_actions}."
        )

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
        filename=f"{game}-action-classifier-{{epoch:03d}}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"{game}/action_classifier",
    )

    if torch.cuda.is_available():
        accelerator = "gpu"

        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        accelerator = "cpu"
        precision = "32-true"

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config["epochs"]),
        precision=precision,
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


# ================================================================
# Skip run 2: load the checkpoint and run one Gym episode
# ================================================================
for x in range(10):
    with (
        skip_run(
            "skip",
            "run_action_classifier_in_gym",
        ) as check,
        check(),
    ):
        runtime_config = dict(config)

        runtime_config["action_classifier_checkpoint"] = (
            "/home/cody/Documents/IHL/eye-world/checkpoints/action_classifier/last.ckpt"
        )

        manager = GymManager(
            config=runtime_config,
            preprocessor_class=RuntimePreprocessor,
            action_net_class=CheckpointActionNet,
            env_name="ALE/MsPacman-v5",
        )

        max_steps = int(
            runtime_config.get(
                "gym_max_steps",
                100_000,
            )
        )

        total_reward = 0.0
        step_count = 0
        final_info = {}

        try:
            state = manager.reset()

            print("\nGym episode started")
            print("Initial state shape:", tuple(state.shape))

            done = False

            while not done and step_count < max_steps:
                state, reward, done, info = manager.step()

                total_reward += float(reward)
                step_count += 1
                final_info = info

            print("\nGym episode finished")
            print("Steps:", step_count)
            print("Total reward:", total_reward)

            if "score" in final_info:
                print("Final score:", final_info["score"])

            if step_count >= max_steps and not done:
                print(f"Episode stopped because it reached gym_max_steps={max_steps}.")

        finally:
            manager.close()


with (
    skip_run(
        "run",
        "run_action_classifier_in_gym",
    ) as check,
    check(),
):
    runtime_config = dict(config)

    runtime_config["action_classifier_checkpoint"] = (
        "/home/cody/Documents/IHL/eye-world/checkpoints/action_classifier/last-v2.ckpt"
    )

    manager = RecordingGymManager(
        config=runtime_config,
        preprocessor_class=RuntimePreprocessor,
        action_net_class=CheckpointActionNet,
        env_name="ALE/MsPacman-v5",
        video_folder=("/home/cody/Documents/IHL/eye-world/videos/action_classifier"),
    )

    max_steps = int(
        runtime_config.get(
            "gym_max_steps",
            100_000,
        )
    )

    total_reward = 0.0
    step_count = 0
    final_info = {}

    try:
        state = manager.reset()

        print("\nGym episode started")
        print("Initial state shape:", tuple(state.shape))

        done = False

        while not done and step_count < max_steps:
            state, reward, done, info = manager.step()

            total_reward += float(reward)
            step_count += 1
            final_info = info

        print("\nGym episode finished")
        print("Steps:", step_count)
        print("Total reward:", total_reward)

        if "score" in final_info:
            print("Final score:", final_info["score"])

        if step_count >= max_steps and not done:
            print(f"Episode stopped because it reached gym_max_steps={max_steps}.")

        if hasattr(manager.action_net, "report"):
            manager.action_net.report()

    finally:
        # RecordVideo writes/finalizes the MP4 when the environment closes.
        manager.close()
