import gymnasium as gym
import torch


class RuntimePreprocessor:
    """Casts a single online observation into the (img, eye_gaze, action)
    sample the offline pipeline expects, then lets the pipeline do the rest.
    `pipeline` is any callable following that sample contract used throughout
    dataset.pre_process — typically a ComposePreprocessor built from
    whichever transforms the target ActionNet was trained with (that set is
    not fixed; pre_process.py is free to grow new ones). No real gaze/action
    labels exist online, so fixed placeholders fill those slots.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.last_action = 0
        self.last_gaze = [(0, 0)]

    def reset(self, obs):
        if hasattr(self.pipeline, "reset"):
            self.pipeline.reset()

        return self.step(obs)

    def step(self, obs):
        sample = (obs, self.last_gaze, self.last_action)

        return self.pipeline(sample)


class RandomActionNet:
    """Samples random actions; useful as a sanity check for GymManager/env wiring."""

    def __init__(self, config, action_space):
        self.action_space = action_space

    def act(self, data_packet):
        return self.action_space.sample()


class TrainedActionNet:
    """Wraps a trained ActionNet checkpoint for online action selection."""

    def __init__(self, config, action_space, action_net):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = action_net

        ckpt_path = config.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            print("No ckpt_path in config; using randomly-initialized ActionNet.")

        self.model.to(self.device)
        self.model.eval()

    def act(self, data_packet):
        img = data_packet[0]
        state = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(state)

        return torch.argmax(logits, dim=1).item()


class GymManager:
    def __init__(
        self,
        config,
        pipeline,
        action_net,
        env_name: str = "ALE/MsPacman-v5",
    ):
        """`pipeline` is the offline preprocessing pipeline (e.g. a
        ComposePreprocessor built from whichever transforms the target
        ActionNet needs — that set is network-specific and not fixed).
        GymManager wraps it in RuntimePreprocessor itself, same as it wraps
        the raw `action_net` model in TrainedActionNet — both wrappers need
        state (action_space, in TrainedActionNet's case) that only exists
        once the env is created below, so callers just hand over the
        building blocks and GymManager assembles them.
        """
        self.env = gym.make(env_name)
        self.preprocessor = RuntimePreprocessor(pipeline)
        self.action_net = TrainedActionNet(config, self.env.action_space, action_net)

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
