import sys
from collections import deque
from pathlib import Path

import ale_py  # ⚠️ force registration of ALE namespace
import gymnasium as gym
import torch
import torch.nn.functional as F
import yaml
from gymnasium.wrappers import RecordVideo

# from config import *
# from gym_manager import GymManager

# from dataset.pre_process import Resize, StackWithLabels
# from models.vjepa import ActionEmbedding, TransformerEncoder, TubeletEmbedding

# import gym

print("Python version:", sys.version)
print("Gym version:", gym.__version__)
# import gymnasium.atari

# Dummy usage to prevent deletion
_ = ale_py.__name__

print("Python version:", sys.version)
print("Gym version:", gym.__version__)


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
        # return torch.cat(list(self.frames), dim=0)


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


from typing import Any, Protocol

import gymnasium as gym


class RandomActionNet:
    def __init__(self, config, action_space):
        self.action_space = action_space

    def act(self, data_packet):
        return self.action_space.sample()


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
    ):
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


####################################################################
# these are debug code to show it works


class RecordingGymManager(GymManager):
    """
    GymManager variant that records episodes as MP4 videos.

    This leaves the original GymManager file unchanged.
    """

    def __init__(
        self,
        config,
        preprocessor_class,
        action_net_class,
        env_name: str = "ALE/MsPacman-v5",
        video_folder: str = "videos/action_classifier",
    ):
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
            episode_trigger=lambda episode_id: True,
            name_prefix="action-classifier",
            disable_logger=False,
        )

        self.preprocessor = preprocessor_class(config)

        self.action_net = action_net_class(
            config=config,
            action_space=self.env.action_space,
        )

        self.state = None

        print("Video directory:", video_path.resolve())


def test():
    config_path = "configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    manager = GymManager(
        config=config,
        preprocessor_class=RuntimePreprocessor,
        action_net_class=RandomActionNet,
    )

    try:
        state = manager.reset()

        print("Initial state")
        print("Shape:", state.shape)
        print("Dtype:", state.dtype)

        total_reward = 0
        done = False
        step_num = 0

        while not done and step_num < 1000:
            state, reward, done, info = manager.step()

            total_reward += reward
            step_num += 1

            print(
                f"step={step_num:4d} "
                f"reward={reward:5.1f} "
                f"done={done} "
                f"state_shape={tuple(state.shape)}"
            )

        print("\nEpisode finished")
        print("Steps:", step_num)
        print("Total reward:", total_reward)

    finally:
        manager.close()


def test_two():
    env = gym.make("ALE/MsPacman-v5")

    print("Gym action_space.n:", env.action_space.n)

    if hasattr(env.unwrapped, "get_action_meanings"):
        print("Gym action meanings:")
        for i, name in enumerate(env.unwrapped.get_action_meanings()):
            print(i, name)
    else:
        print("No get_action_meanings found.")


test()
