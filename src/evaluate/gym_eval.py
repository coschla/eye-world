from pathlib import Path
from typing import Any

import ale_py  # noqa: F401  # force registration of the ALE namespace
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

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

        # -----------------------------------------------------
        # SPACE INVADERS SHOULD HAVE SIX ACTIONS
        # -----------------------------------------------------

        expected_actions = [
            "NOOP",
            "FIRE",
            "RIGHT",
            "LEFT",
            "RIGHTFIRE",
            "LEFTFIRE",
        ]

        print(
            "Gym action meanings:",
            action_meanings,
        )

        if action_space.n != 6:
            raise RuntimeError(
                f"Expected Space Invaders to expose 6 actions, "
                f"but Gym reports {action_space.n}."
            )

        if list(action_meanings) != expected_actions:
            raise RuntimeError(
                "Unexpected Space Invaders action mapping.\n"
                f"Expected: {expected_actions}\n"
                f"Received: {list(action_meanings)}"
            )

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
            return torch.load(
                checkpoint_path,
                map_location=self.device,
            )

    def _extract_action_net_state(
        self,
        checkpoint: dict,
    ) -> dict:

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

        expected_state = self.model.state_dict()

        unexpected_shapes = []

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
                f"Checkpoint architecture does not match ActionNet.\n{shape_text}"
            )

        return extracted_state

    def act(
        self,
        data_packet: Any,
    ) -> int:

        state = torch.as_tensor(
            data_packet,
            dtype=torch.float32,
            device=self.device,
        )

        # Four RGB frames stacked along the channel dimension:
        #
        # [12, 84, 84]
        # ->
        # [1, 12, 84, 84]
        if state.ndim == 3:
            state = state.unsqueeze(0)

        if state.ndim != 4:
            raise ValueError(
                f"ActionNet expected [batch, 12, H, W], got {tuple(state.shape)}"
            )

        if state.shape[1] != 12:
            raise ValueError(
                "ActionNet expected four RGB frames "
                "(12 channels), "
                f"got {tuple(state.shape)}"
            )

        with torch.inference_mode():
            logits = self.model(state)

        # One action prediction for the four-frame stack.
        # Expected: [B, 1, 6]
        if logits.ndim != 3:
            raise RuntimeError(
                f"Expected ActionNet output [B, 1, 6], got {tuple(logits.shape)}"
            )

        if logits.shape[-1] != 6:
            raise RuntimeError(
                f"Expected six Space Invaders classes, got {tuple(logits.shape)}"
            )

        action_logits = logits[:, -1, :]

        action = int(action_logits.argmax(dim=-1).item())

        return action


class GymManager:
    def __init__(
        self,
        config,
        preprocessor_pipeline,
        action_net,
        env_name: str = "ALE/SpaceInvaders-v5",
        record_video: bool = False,
        video_folder: str = "videos/action_classifier",
        episode_trigger=lambda episode_id: True,
    ):
        if record_video:
            video_path = Path(video_folder)
            video_path.mkdir(
                parents=True,
                exist_ok=True,
            )

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

            print(
                "Video directory:",
                video_path.resolve(),
            )

        else:
            self.env = gym.make(env_name)

        action_meanings = self.env.unwrapped.get_action_meanings()

        print(
            "Environment actions:",
            action_meanings,
        )

        self.preprocessor = RuntimePreprocessor(preprocessor_pipeline)

        self.action_net = RuntimeActionNet(
            action_net,
            config=config,
            action_space=self.env.action_space,
            action_meanings=action_meanings,
        )

        self.state = None

        # -----------------------------------------------------
        # LIFE TRACKING
        # -----------------------------------------------------
        self.lives = None

    def reset(self):
        observation, info = self.env.reset()

        # -----------------------------------------------------
        # TRACK STARTING NUMBER OF LIVES
        # -----------------------------------------------------
        self.lives = info.get("lives")

        print(
            "Starting lives:",
            self.lives,
        )

        self.state = self.preprocessor.reset(observation)

        return self.state

    def step(self):
        # -----------------------------------------------------
        # MODEL SELECTS ACTION
        # -----------------------------------------------------

        action = self.action_net.act(self.state)

        # -----------------------------------------------------
        # EXECUTE EXACTLY THAT ACTION
        # -----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        done = terminated or truncated

        # -----------------------------------------------------
        # LIFE TRACKING
        # -----------------------------------------------------

        current_lives = info.get("lives")

        life_lost = (
            not done
            and self.lives is not None
            and current_lives is not None
            and current_lives < self.lives
        )

        if life_lost:
            print(f"Life lost: {self.lives} -> {current_lives}")

        # Store the latest life count.
        if current_lives is not None:
            self.lives = current_lives

        # -----------------------------------------------------
        # NEXT STATE
        # -----------------------------------------------------

        self.state = self.preprocessor.step(observation)

        return (
            self.state,
            reward,
            done,
            info,
        )

    def close(self):
        self.env.close()
