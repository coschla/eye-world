import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn

from dataset.pre_process import ComposePreprocessor, Resize, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


class ActionNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # Atari style input
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # Fully connected action head
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),  # 3136
            nn.LeakyReLU(),
            nn.Linear(512, num_actions),
        )  # 1568

    def forward(self, x):
        x = self.conv_layers(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        x = self.fc_layers(x)

        return x  # logits


# from when we wanted a concolutional action network
with skip_run("skip", "action_condition_classifier") as check, check():
    game = config["games"][0]  # ONLY ONE GAME

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])

    logger = TensorBoardLogger("tb_logs", name=f"{game}/action_classifier/")

    # -----------------------------
    # Single dataset + dataloader
    # -----------------------------
    ds = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    train_loader = ds["train"]
    test_loader = ds["test"]
    loader = ds["train"]
    batch = next(iter(loader))

    _, _, actions = batch

    data_loaders = {"train": train_loader, "test": test_loader}

    # -----------------------------
    # Model
    # -----------------------------
    num_actions = config.get("num_actions", 18)
    net = ActionNet(num_actions)

    model = ActionTraining(hparams=config, net=net, data_loader=data_loaders)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="cpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model)

    class RuntimePreprocessor:
        def __init__(self, config):
            self.pipeline = ComposePreprocessor(
                [
                    Resize(config),
                    StackWithLabels(config),
                ]
            )

            # dummy placeholders (since env doesn't provide these)
            self.last_action = 0
            self.last_gaze = [(0, 0)]  # or whatever shape your code expects

        def step(self, obs):
            sample = (obs, self.last_gaze, self.last_action)

            stacked_img, stacked_gaze, stacked_action = self.pipeline(sample)

            # update last action (optional, but keeps things consistent)
            self.last_action = stacked_action[-1].item()

            return stacked_img

    # =========================
    # Load model
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_actions = 18  # MsPacman action space
    # model = ActionNet(num_actions)
    """
        checkpoint_path = "your_checkpoint.ckpt"  # <-- CHANGE THIS
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # If saved with PyTorch Lightning
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        """
    model.to(device)
    model.eval()

    # =========================
    # Setup env + video
    # =========================
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # record every episode
    )

    runtime_preprocessor = RuntimePreprocessor(config)

    obs, info = env.reset()
    state = runtime_preprocessor.step(obs)

    done = False
    total_reward = 0
    final_score = 0  # for Atari score if available

    # =========================
    # Run agent
    # =========================
    while not done:
        state_tensor = state.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = runtime_preprocessor.step(obs)
        total_reward += reward

        # Some Atari envs include score in info
        if "score" in info:
            final_score = info["score"]

    # =========================
    # Results
    # =========================
    print("Total reward:", total_reward)
    print("Final score:", final_score)

    env.close()
"""


class ActionNet(nn.Module):
    def __init__(self, num_actions, num_future=2):
        super(ActionNet, self).__init__()

        self.num_future = num_future
        self.num_actions = num_actions

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),  # ✅ fixed bug
            nn.LeakyReLU(),
        )

        # Fully connected head
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_actions * num_future),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.fc_layers(x)
        logits = logits.view(-1, self.num_future, self.num_actions)  # [B, 2, A]

        return logits
"""
