from collections import deque
from pathlib import Path
from typing import Any, Protocol

import ale_py  # noqa: F401  # force registration of the ALE namespace
import gymnasium as gym
import torch
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo


class Resize:
    def __init__(self, config):
        self.size = 84

    def __call__(self, sample):
        obs, _, _ = sample

        obs = torch.tensor(obs, dtype=torch.float32) / 255.0

        if obs.ndim == 3:
            obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]

        obs = obs.unsqueeze(0).unsqueeze(0)
        obs = F.interpolate(obs, size=(self.size, self.size), mode="bilinear")
        obs = obs.squeeze(0)

        return obs, None, None


class StackWithLabels:
    def __init__(self, stack_size=4):
        self.frames = deque(maxlen=stack_size)
        self.stack_size = stack_size

    def reset(self, first_frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)
        return self._get_stack()

    def step(self, frame):
        self.frames.append(frame)
        return self._get_stack()

    def _get_stack(self):
        return torch.stack(list(self.frames), dim=0)


class RuntimePreprocessor:
    def __init__(self, config):
        self.resize = Resize(config)
        self.stack = StackWithLabels(4)

    def reset(self, obs):
        obs, _, _ = self.resize((obs, None, None))
        return self.stack.reset(obs)

    def step(self, obs):
        obs, _, _ = self.resize((obs, None, None))
        return self.stack.step(obs)


class Preprocessor(Protocol):
    """Interface every runtime preprocessor must implement."""

    def reset(self, observation: Any) -> Any: ...

    def step(self, observation: Any) -> Any: ...


class ActionNet(Protocol):
    """Interface every action-selection class must implement."""

    def act(self, data_packet: Any) -> int: ...


class GymManager:
    def __init__(
        self,
        config,
        preprocessor_class,
        action_net_class,
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

        self.preprocessor = preprocessor_class(config)
        self.action_net = action_net_class(
            config=config,
            action_space=self.env.action_space,
        )

        self.state = None

    def reset(self):
        observation, info = self.env.reset()

        self.state = self.preprocessor.reset(observation)

        return self.state

    def step(self):
        # ActionNet receives the packet created by the preprocessor.
        action = self.action_net.act(self.state)

        observation, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated
        self.state = self.preprocessor.step(observation)

        return self.state, reward, done, info

    def close(self):
        self.env.close()
