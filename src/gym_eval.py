import sys
from collections import deque

import ale_py  # ⚠️ force registration of ALE namespace
import gymnasium as gym
import torch
import torch.nn.functional as F

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


class GymManager:
    def __init__(
        self,
        config,
        env_name="ALE/MsPacman-v5",
    ):

        self.env = gym.make(env_name)

        self.preprocessor = RuntimePreprocessor(config)

    def reset(self):

        obs, _ = self.env.reset()

        return self.preprocessor.reset(obs)

    def step(self, action):

        obs, reward, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated

        state = self.preprocessor.step(obs)

        return state, reward, done


####################################################################
# these are debug code to show it works
def test():
    manager = GymManager(config=None)

    state = manager.reset()

    print("Initial state")
    print("Shape:", state.shape)
    print("Dtype:", state.dtype)

    total_reward = 0
    done = False
    step_num = 0

    while not done and step_num < 1000:
        action = manager.env.action_space.sample()

        state, reward, done = manager.step(action)

        total_reward += reward
        step_num += 1

        print(
            f"step={step_num:4d} "
            f"action={action:2d} "
            f"reward={reward:5.1f} "
            f"done={done} "
            f"state_shape={tuple(state.shape)}"
        )

    print("\nEpisode finished")
    print("Steps:", step_num)
    print("Total reward:", total_reward)


def test_two():
    env = gym.make("ALE/MsPacman-v5")

    print("Gym action_space.n:", env.action_space.n)

    if hasattr(env.unwrapped, "get_action_meanings"):
        print("Gym action meanings:")
        for i, name in enumerate(env.unwrapped.get_action_meanings()):
            print(i, name)
    else:
        print("No get_action_meanings found.")
