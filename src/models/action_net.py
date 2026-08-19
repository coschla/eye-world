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

# ============================================================
import sys
from pathlib import Path

from vjepa import ActionEmbedding, TransformerEncoder, TubeletEmbedding

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
    def __init__(
        self,
        num_actions: int = 9,
        num_predictions: int = 4,
        token_count: int = 16,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.num_predictions = num_predictions
        self.token_count = token_count
        self.embed_dim = embed_dim

        combined_size = token_count * embed_dim * 2

        self.present_norm = nn.LayerNorm(embed_dim)
        self.future_norm = nn.LayerNorm(embed_dim)

        self.action_head = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                num_predictions * num_actions,
            ),
        )

    def forward(
        self,
        present_latent: torch.Tensor,
        future_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            present_latent: [B, 16, 768]
            future_latent:  [B, 16, 768]

        Output:
            logits: [B, 4, 9]
        """
        expected_shape = (
            self.token_count,
            self.embed_dim,
        )

        if present_latent.shape[1:] != expected_shape:
            raise ValueError(
                "Incorrect present-latent shape. "
                f"Expected [B, 16, 768], received "
                f"{tuple(present_latent.shape)}."
            )

        if future_latent.shape[1:] != expected_shape:
            raise ValueError(
                "Incorrect future-latent shape. "
                f"Expected [B, 16, 768], received "
                f"{tuple(future_latent.shape)}."
            )

        present_latent = self.present_norm(present_latent)

        future_latent = self.future_norm(future_latent)

        # Preserve all 16 tubelets from both latents.
        combined = torch.cat(
            [
                present_latent,
                future_latent,
            ],
            dim=-1,
        )

        # [B, 16, 1536] -> [B, 24576]
        combined = combined.flatten(start_dim=1)

        logits = self.action_head(combined)

        # [B, 36] -> [B, 4, 9]
        logits = logits.reshape(
            logits.shape[0],
            self.num_predictions,
            self.num_actions,
        )

        return logits


class VJEPALatentExtractor(nn.Module):
    def __init__(self, checkpoint_path, config, device):
        super().__init__()

        self.device = torch.device(device)
        self.embed_dim = 768

        patch_dim = 1 if config.get("grey_scale_v", True) else 3

        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=self.embed_dim,
            img_size=config.get("image_size", 84),
        )

        self.student = TransformerEncoder(
            dim=self.embed_dim,
            depth=12,
            heads=12,
            mlp_dim=3072,
        )

        self.action_embed = ActionEmbedding(
            num_actions=18,
            token_dim=self.embed_dim,
        )

        self.predictor = nn.Transformer(
            d_model=self.embed_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint["state_dict"]

        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

        self.tubelet_embed.load_state_dict(
            {
                key.removeprefix("tubelet_embed."): value
                for key, value in state_dict.items()
                if key.startswith("tubelet_embed.")
            },
            strict=True,
        )

        self.student.load_state_dict(
            {
                key.removeprefix("student."): value
                for key, value in state_dict.items()
                if key.startswith("student.")
            },
            strict=True,
        )

        self.action_embed.load_state_dict(
            {
                key.removeprefix("action_embed."): value
                for key, value in state_dict.items()
                if key.startswith("action_embed.")
            },
            strict=False,
        )

        self.predictor.load_state_dict(
            {
                key.removeprefix("latent_predictor."): value
                for key, value in state_dict.items()
                if key.startswith("latent_predictor.")
            },
            strict=False,
        )

        self.to(self.device)
        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad = False


# ================================================================
# Lightning training module
# ================================================================
class ActionTraining(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        net: nn.Module,
        latent_extractor: VJEPALatentExtractor,
        data_loader: dict,
    ):
        super().__init__()

        self.net = net
        self.latent_extractor = latent_extractor
        self.data_loaders = data_loader

        self.learning_rate = float(hparams.get("learning_rate", 1e-3))

        self.weight_decay = float(hparams.get("weight_decay", 0.0))

        # The V-JEPA model stays frozen.
        self.latent_extractor.eval()

        for parameter in self.latent_extractor.parameters():
            parameter.requires_grad = False

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    @torch.no_grad()
    def forward(
        self,
        present_frames: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Input:
            Four present frames.

        Output:
            Present latent and predicted future latent.
        """
        present_latent = self.encode_present(present_frames)

        predicted_future_latent = self.predict_future_latent(
            present_latent=present_latent,
            actions=actions,
        )

        return {
            "present_latent": present_latent,
            "predicted_future_latent": predicted_future_latent,
            "present_vector": present_latent.mean(dim=1),
            "predicted_future_vector": predicted_future_latent.mean(dim=1),
        }

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

    def encode_present(
        self,
        present_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            Four present frames.

        Expected shape depends on TubeletEmbedding, typically:

            [batch, time, channels, height, width]

        For grayscale:
            [batch, 4, 1, 84, 84]

        Output:
            Present latent tokens:
            [batch, present_tokens, 768]
        """
        present_frames = present_frames.to(
            device=self.device,
            dtype=torch.float32,
        )

        present_tokens = self.tubelet_embed(present_frames)
        present_latent = self.student(present_tokens)

        return present_latent

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        images, _, actions = batch

        # [B, 4, 1, 84, 84]
        present_frames = self._prepare_images(images)

        # [B, 4]
        targets = self._prepare_actions(actions)

        # Extract present and predicted-future latents from
        # only the four present frames.
        with torch.no_grad():
            latent_output = self.latent_extractor(
                present_frames=present_frames,
            )

        present_latent = latent_output["present_latent"]

        predicted_future_latent = latent_output["predicted_future_latent"]

        # Print only the first training batch.
        if stage == "train" and self.global_step == 0:
            print(
                "Present frames:",
                tuple(present_frames.shape),
            )
            print(
                "Present latent:",
                tuple(present_latent.shape),
            )
            print(
                "Predicted future latent:",
                tuple(predicted_future_latent.shape),
            )
            print(
                "Targets:",
                tuple(targets.shape),
            )

        # [B, 4, 9]
        logits = self.net(
            present_latent,
            predicted_future_latent,
        )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}."
            )

        if logits.shape[-1] != 9:
            raise ValueError(
                f"Expected nine Gym action classes, but received {tuple(logits.shape)}."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, 9),
            targets.reshape(-1),
        )

        predicted_actions = logits.argmax(dim=-1)

        accuracy = (predicted_actions == targets).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )

        return loss

    def predict_future_latent(
        self,
        present_latent: torch.Tensor,
        actions: torch.Tensor | None = None,
        future_token_count: int | None = None,
    ) -> torch.Tensor:
        """
        Predict the latent representation of the four future frames.

        No future images or true future latent are provided.
        """
        source = present_latent

        if actions is not None:
            actions = actions.to(
                device=self.device,
                dtype=torch.long,
            )

            action_tokens = self.action_embed(actions)

            if action_tokens.ndim == 2:
                action_tokens = action_tokens.unsqueeze(1)

            source = torch.cat(
                [source, action_tokens],
                dim=1,
            )

        if future_token_count is None:
            # Use this only when the future and present groups produce
            # the same number of latent tokens.
            future_token_count = present_latent.shape[1]

        future_queries = torch.zeros(
            present_latent.shape[0],
            future_token_count,
            self.embed_dim,
            device=self.device,
            dtype=present_latent.dtype,
        )

        predicted_future_latent = self.predictor(
            src=source,
            tgt=future_queries,
        )

        return predicted_future_latent

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
        "run",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_extractor = VJEPALatentExtractor(
        checkpoint_path=(
            "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_0/checkpoints/epoch=49-step=164350.ckpt"
        ),
        config=config,
        device=device,
    )

    action_network = ActionNet(
        num_actions=9,
        num_predictions=4,
        token_count=16,
        embed_dim=768,
    )

    training_model = ActionTraining(
        hparams=config,
        net=action_network,
        latent_extractor=latent_extractor,
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
        "skip",
        "run_action_classifier_in_gym",
    ) as check,
    check(),
):
    runtime_config = dict(config)

    runtime_config["action_classifier_checkpoint"] = (
        "/home/cody/Documents/IHL/eye-world/checkpoints/action_classifier/last.ckpt"
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
