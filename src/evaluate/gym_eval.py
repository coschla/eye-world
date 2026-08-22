from pathlib import Path
from typing import Any

import ale_py  # noqa: F401  # force registration of the ALE namespace
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from models.utils import map_canonical_actions

# ActionNet only consumes the stacked image, never gaze/action, so these
# placeholders just need to satisfy the caller's pipeline's sample shape.
# eye_gaze_to_density_image expects a sequence of (x, y) points per frame.
_DUMMY_GAZE = [(0, 0)]
_DUMMY_ACTION = 0


class RuntimePreprocessor:
    """
    Wraps the caller's own preprocessing pipeline instance (e.g. the same
    ComposePreprocessor of Resize, StackWithLabels, or whatever else they
    compose, used to build offline training data), so frames seen during
    Gym evaluation are processed identically to training data.

    Some steps (e.g. StackWithLabels) hold per-episode state, like a
    frame deque, that must not leak across episodes. Every preprocessor
    implements reset(), so ComposePreprocessor.reset() clears them all.
    """

    def __init__(self, preprocessor_pipeline):
        self.preprocessor_pipeline = preprocessor_pipeline

    def _process(self, obs):
        stacked_img, _, _ = self.preprocessor_pipeline(
            (obs, _DUMMY_GAZE, _DUMMY_ACTION)
        )

        return stacked_img

    def reset(self, obs):
        self.preprocessor_pipeline.reset()

        return self._process(obs)

    def step(self, obs):
        return self._process(obs)


class RuntimeActionNet:
    """
    Load a trained ActionNet and provide GymManager's required:

        act(data_packet) -> int

    The network was trained on ActionNet's fixed 9-way canonical action
    space (see atari_to_gym / CANONICAL_ACTION_NAMES in models.utils),
    not directly on the current Gym env's action IDs. Since different
    games expose different action sets (e.g. Breakout has 4 actions,
    MsPacman has 9), predictions are translated per-env via
    map_canonical_actions() using action_meanings.
    """

    def __init__(
        self,
        action_net,
        config: dict,
        action_space: Any,
        action_meanings: list,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(config["action_classifier_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Action-classifier checkpoint was not found: {checkpoint_path}"
            )

        self.action_mapping = map_canonical_actions(action_meanings)
        self.model = action_net

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
            canonical_action = int(predicted_actions[0, -1].item())

        return self.action_mapping[canonical_action]


class GymManager:
    def __init__(
        self,
        config,
        preprocessor_pipeline,
        action_net,
        env_name: str = "ALE/MsPacman-v5",
        record_video: bool = False,
        video_folder: str = "videos/action_classifier",
        episode_trigger=lambda episode_id: True,
    ):
        if record_video:
            video_path = Path(video_folder)
            video_path.mkdir(parents=True, exist_ok=True)

            # Atari video recording requires render_mode="rgb_array".
            base_env = gym.make(
                env_name,
                render_mode="rgb_array",
            )

            self.env = RecordVideo(
                base_env,
                video_folder=str(video_path),
                episode_trigger=episode_trigger,
                name_prefix="action-classifier",
                disable_logger=False,
            )

            print("Video directory:", video_path.resolve())
        else:
            self.env = gym.make(env_name)

        action_meanings = self.env.unwrapped.get_action_meanings()

        self.preprocessor = RuntimePreprocessor(preprocessor_pipeline)
        self.action_net = RuntimeActionNet(
            action_net,
            config=config,
            action_space=self.env.action_space,
            action_meanings=action_meanings,
        )

        # ActionNet's canonical action space (see atari_to_gym) collapses
        # every *FIRE action into base movement, so it can never predict
        # FIRE on its own. In games that require FIRE to launch/serve
        # (Breakout, Pong, ...), the game would otherwise sit frozen
        # forever. Auto-fire on reset and after every life lost, same as
        # OpenAI Baselines' FireResetEnv.
        self.fire_action_id = (
            action_meanings.index("FIRE") if "FIRE" in action_meanings else None
        )

        self.state = None
        self.lives = None

    def _auto_fire(self, observation, info):
        if self.fire_action_id is None:
            return observation, info

        observation, _, terminated, truncated, info = self.env.step(self.fire_action_id)

        if terminated or truncated:
            observation, info = self.env.reset()

        return observation, info

    def reset(self):
        observation, info = self.env.reset()
        observation, info = self._auto_fire(observation, info)

        self.lives = info.get("lives")
        self.state = self.preprocessor.reset(observation)

        return self.state

    def step(self):
        # ActionNet receives the packet created by the preprocessor.
        action = self.action_net.act(self.state)

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        lives = info.get("lives")

        if not done and self.lives is not None and lives is not None and lives < self.lives:
            observation, info = self._auto_fire(observation, info)

        self.lives = info.get("lives", lives)
        self.state = self.preprocessor.step(observation)

        return self.state, reward, done, info

    def close(self):
        self.env.close()
