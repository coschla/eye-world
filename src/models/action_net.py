from pathlib import Path
from typing import Any

import torch
from torch import nn

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
