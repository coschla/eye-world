# ============================================================
# [SECTION 1] IMPORTS + FIXED MODEL IMPORTS (IMPORTANT FIX)
# ============================================================

import os
import random
import sys
from collections import deque

import ale_py  # ⚠️ force registration of ALE namespace
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from vjepa import ActionEmbedding, TransformerEncoder, TubeletEmbedding

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# [SECTION 2] LOGGING SETUP
# ============================================================

base_dir = "tb_logs"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [
    d
    for d in os.listdir(base_dir)
    if d.startswith("DQN_") and os.path.isdir(os.path.join(base_dir, d))
]

run_numbers = []
for d in existing_runs:
    try:
        run_numbers.append(int(d.split("_")[1]))
    except:
        pass

next_run = max(run_numbers) + 1 if run_numbers else 1
log_dir = os.path.join(base_dir, f"DQN_{next_run}")

writer = SummaryWriter(log_dir=log_dir)
print(f"[TensorBoard] Logging to: {log_dir}")


# ============================================================
# [SECTION 3] FIXED VJEPA LATENT EXTRACTOR (REAL MODEL WRAPPER)
# ============================================================
import torch


class VJEPALatentExtractor(nn.Module):
    def __init__(self, checkpoint_path, config, device):
        super().__init__()

        # -------------------------
        # Model config (must match training)
        # -------------------------
        patch_dim = 1 if config.get("grey_scale_v", True) else 3
        embed_dim = 768
        heads = 12
        mlp_dim = 3072

        # -------------------------
        # Tubelet embedding
        # -------------------------
        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("image_size", 84),
        )

        # -------------------------
        # Student encoder
        # -------------------------
        self.student = TransformerEncoder(
            dim=embed_dim,
            depth=12,
            heads=heads,
            mlp_dim=mlp_dim,
        )

        # -------------------------
        # Action embedding
        # -------------------------
        self.action_embed = ActionEmbedding(
            num_actions=18,
            token_dim=embed_dim,
        )

        # -------------------------
        # Predictor (same as training)
        # -------------------------
        self.predictor = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )

        # -------------------------
        # Load checkpoint (LIGHTNING FORMAT)
        # -------------------------
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]

        # Strip Lightning "model." prefix
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        # -------------------------
        # Load submodules safely
        # -------------------------
        self.tubelet_embed.load_state_dict(
            {
                k.replace("tubelet_embed.", ""): v
                for k, v in state_dict.items()
                if k.startswith("tubelet_embed.")
            },
            strict=True,
        )

        self.student.load_state_dict(
            {
                k.replace("student.", ""): v
                for k, v in state_dict.items()
                if k.startswith("student.")
            },
            strict=True,
        )

        self.action_embed.load_state_dict(
            {
                k.replace("action_embed.", ""): v
                for k, v in state_dict.items()
                if k.startswith("action_embed.")
            },
            strict=False,
        )

        self.predictor.load_state_dict(
            {
                k.replace("latent_predictor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("latent_predictor.")
            },
            strict=False,
        )

        self.to(device)
        self.eval()

    # -------------------------------------------------
    # FORWARD PASS (matches training exactly)
    # -------------------------------------------------
    @torch.no_grad()
    def forward(self, frames, action=torch.tensor([2])):
        """
        frames: [B, T, 1, H, W]   (context frames only)
        action: [B]              (last observed action)

        returns: [B, D] predicted future latent
        """
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action = action.to(DEVICE).long()

        # -------------------------
        # 1. Tubelet embedding
        # -------------------------
        tokens = self.tubelet_embed(frames)  # [B, N, D]

        # -------------------------
        # 2. Student encoder
        # -------------------------
        tokens = self.student(tokens)  # [B, N, D]

        # -------------------------
        # 3. Action embedding
        # -------------------------
        #
        # action_token = self.action_embed(action)  # [B, D]
        # action_token = self.action_embed(action).unsqueeze(1)

        # action_token = action_token.unsqueeze(1)  # [B, 1, D]
        # -------------------------
        # Action embedding (FIXED)
        # -------------------------
        action_token = self.action_embed(action)  # [B, D]

        # FORCE sequence dimension
        if action_token.dim() == 2:
            action_token = action_token.unsqueeze(1)  # [B, 1, D]
        # -------------------------
        # 4. Build sequence (same as training)
        # -------------------------
        seq = torch.cat([tokens, action_token], dim=1)  # [B, N+1, D]

        # -------------------------
        # 5. Predictor (causal transformer)
        # -------------------------
        pred_seq = self.predictor(seq[:, :-1], seq[:, :-1])

        # -------------------------
        # 6. Output latent (next-step prediction)
        # -------------------------
        pred_latent = pred_seq  # [:, -1, :]  # [B, D]

        return pred_latent


'''
class VJEPALatentExtractor(nn.Module):
    def __init__(self, checkpoint_path, config, device):
        super().__init__()

        patch_dim = 1 if config.get("grey_scale_v", True) else 3
        embed_dim = 768
        heads = 12
        mlp_dim = 3072

        # -------------------------
        # Core modules
        # -------------------------
        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("image_size", 84),
        )

        self.student = TransformerEncoder(
            dim=embed_dim,
            depth=12,
            heads=heads,
            mlp_dim=mlp_dim,
        )

        self.action_embed = ActionEmbedding(
            num_actions=18,
            token_dim=embed_dim,
        )

        self.predictor = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )

        # -------------------------
        # Load checkpoint
        # -------------------------

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        student_keys = [k for k in state_dict.keys() if k.startswith("model.student.")]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        for k in list(state_dict.keys())[:30]:
            print(k)
        # Load each part EXACTLY like training
        """
        self.student.load_state_dict(
            {
                k.replace("model.student.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.student.")
            }
        )
        """
        self.tubelet_embed.load_state_dict(
            {
                k.replace("model.tubelet_embed.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.tubelet_embed.")
            }
        )

        self.action_embed.load_state_dict(
            {
                k.replace("action_embed.", ""): v
                for k, v in state_dict.items()
                if k.startswith("action_embed.")
            },
            strict=False,
        )

        self.predictor.load_state_dict(
            {
                k.replace("latent_predictor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("latent_predictor.")
            },
            strict=False,
        )

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, frames, action):
        """
        frames: [B, T, 1, H, W]  (context frames ONLY)
        action: [B]              (last action)

        returns: [B, D] predicted future latent
        """

        # -------------------------
        # Student encoding
        # -------------------------
        tokens = self.tubelet_embed(frames)  # [B, N, D]
        # tokens = self.student(tokens)  # [B, N, D]

        # -------------------------
        # Action embedding
        # -------------------------
        action_token = self.action_embed(action)  # [B, D]
        action_token = action_token.unsqueeze(1)  # [B, 1, D]

        # -------------------------
        # Sequence construction
        # -------------------------
        seq = torch.cat([tokens, action_token], dim=1)  # [B, N+1, D]

        # -------------------------
        # Predictor (same as training)
        # -------------------------
        pred_seq = self.predictor(seq[:, :-1], seq[:, :-1])

        pred_latent = pred_seq[:, -1, :]  # [B, D]

        return pred_latent
'''

'''

class VJEPALatentExtractor(nn.Module):
    def __init__(self, checkpoint_path, config, device):
        super().__init__()

        patch_dim = 1 if config.get("grey_scale_v", True) else 3
        embed_dim = 768
        heads = 12
        mlp_dim = 3072

        # -------------------------
        # Build SAME architecture as training
        # -------------------------
        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("image_size", 84),  # ✅ FIXED BUG HERE
        )

        self.encoder = TransformerEncoder(
            dim=embed_dim,
            depth=12,
            heads=heads,
            mlp_dim=mlp_dim,
        )

        # -------------------------
        # Load checkpoint safely
        # -------------------------
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = {
            k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
        }

        self.load_state_dict(state_dict, strict=False)

        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, frames):
        """
        frames: [B, 4, 1, 84, 84]
        returns: [B, 768]
        """

        tokens = self.tubelet_embed(frames)
        tokens = self.encoder(tokens)
        return tokens
        # return tokens.mean(dim=1)  # pooling
'''

# ============================================================
# [SECTION 4] HYPERPARAMETERS
# ============================================================

"""GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 5_000
TARGET_UPDATE = 5000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 200_000
"""
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 10_000
TARGET_UPDATE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.08
EPSILON_DECAY = 300_000

# ============================================================
# [SECTION 5] CONFIG
# ============================================================
"""
config = {
    "image_size": 84,
    "context_frames": 4,
}
"""

config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
# ============================================================
# [SECTION 6] PREPROCESSING
# ============================================================


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


# ============================================================
# [SECTION 7] DQN MODEL
# ============================================================
"""

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        t, c, h, w = input_shape

        in_channels = t * c  # ✅ THIS IS THE FIX

        # print(f"[DQN INIT] input_shape={input_shape}")
        # print(f"[DQN INIT] in_channels={in_channels}")

        self.conv = nn.Sequential(
            nn.Conv2d(t * c, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            # print(f"[DQN INIT] dummy shape: {dummy.shape}")

            conv_out = self.conv(dummy)
            # print(f"[DQN INIT] conv_out shape: {conv_out.shape}")

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            # dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_out + 768, 512),  # latent dim fixed
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x, latent):
        x = self.conv(x)
        x = torch.cat([x, latent], dim=1)
        return self.head(x)"""

"""
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        t, c, h, w = input_shape
        in_channels = t * c

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            conv_out = self.conv(dummy).shape[1]

        # ✅ NEW: token projection (Option 2)
        self.token_proj = nn.Sequential(
            nn.Flatten(),  # [B, 16 * 768]
            nn.Linear(16 * 768, 8 * 16),
            nn.ReLU(),
            nn.Linear(8 * 768, 2 * 768),
            nn.ReLU(),
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out + 768, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x, latent):
        x = self.conv(x)

        # ✅ NEW: project tokens
        latent = self.token_proj(latent)

        x = torch.cat([x, latent], dim=1)
        return self.head(x)
"""


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, latent_dim=768):
        super().__init__()

        t, c, h, w = input_shape
        in_channels = t * c

        # -------------------------
        # CNN encoder for pixels
        # -------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            conv_out = self.conv(dummy).shape[1]

        # -------------------------
        # JEPA latent projector
        # -------------------------
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # -------------------------
        # final Q head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(conv_out + 256, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x, latent):
        x = self.conv(x)
        latent = self.token_proj(latent)
        x = torch.cat([x, latent], dim=1)
        return self.head(x)


'''
    def forward(self, x, latent):
        """
        x:      [B, T*C, H, W]
        latent: [B, D]  (IMPORTANT: already pooled)
        """

        x = self.conv(x)

        # safety: ensure correct shape
        if latent.dim() == 3:
            latent = latent.mean(dim=1)  # [B, D]

        latent = self.token_proj(latent)

        x = torch.cat([x, latent], dim=1)

        return self.head(x)'''


# ============================================================
# [SECTION 8] REPLAY BUFFER
# ============================================================


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s2, d, l, l2):
        self.buffer.append((s, a, r, s2, d, l, l2))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, l, l2 = zip(*batch)

        return (
            torch.stack(s).to(DEVICE),
            torch.tensor(a, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.stack(s2).to(DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
            torch.stack(l).to(DEVICE),
            torch.stack(l2).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# [SECTION 9] EPSILON
# ============================================================


def get_epsilon(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY)


# ============================================================
# [SECTION 10] TRAIN STEP
# ============================================================


def train_step(policy, target, buffer, optim):
    if len(buffer) < MIN_BUFFER_SIZE:
        return None

    s, a, r, s2, d, l, l2 = buffer.sample(BATCH_SIZE)

    B, T, C, H, W = s.shape
    s = s.view(B, T * C, H, W)
    s2 = s2.view(B, T * C, H, W)

    q = policy(s, l).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = policy(s2, l2).argmax(1, keepdim=True)
        next_q = target(s2, l2).gather(1, next_actions).squeeze(1)
        target_q = r + GAMMA * next_q * (1 - d)
    # with torch.no_grad():
    # next_q = target(s2, l2).max(1)[0]
    # target_q = r + GAMMA * next_q * (1 - d)

    loss = F.smooth_l1_loss(q, target_q)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()


# ============================================================
# [SECTION 11] TRAINING LOOP (FIXED JEPA INTEGRATION)
# ============================================================


def train():
    env = gym.make("ALE/MsPacman-v5")
    pre = RuntimePreprocessor(config)

    # -------------------------
    # REAL JEPA LOADING (FIXED)
    # -------------------------
    jepa = VJEPALatentExtractor(
        checkpoint_path="/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_world_model/version_9/checkpoints/epoch=49-step=164350.ckpt",
        config=config,
        device=DEVICE,
    )

    obs, _ = env.reset()
    state = pre.reset(obs)

    policy = DQN(state.shape, env.action_space.n).to(DEVICE)
    target = DQN(state.shape, env.action_space.n).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optim = torch.optim.Adam(policy.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    step = 0
    episode = 0
    while True:
        obs, _ = env.reset()
        state = pre.reset(obs)

        done = False
        total = 0

        while not done:
            eps = get_epsilon(step)

            # -------------------------
            # raw state for JEPA
            # -------------------------
            s_raw = state.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

            # -------------------------
            # flat state for DQN
            # -------------------------
            s_flat = s_raw.view(1, -1, 84, 84)  # (1, T*C, H, W)

            with torch.no_grad():
                lat = jepa(s_raw).mean(dim=1)  # [B, 768]

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy(s_flat, lat)
                    # q = policy(torch.cat([s_flat.view(1, -1), lat], dim=-1))
                    action = torch.argmax(q, dim=1).item()

            """
            with torch.no_grad():
                lat = jepa(s_raw)  # [B, T, D]
                lat = lat.mean(dim=1)  # [B, D]
            """

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy(s_flat, lat)
                    # q = policy(torch.cat([s_flat.view(1, -1), lat], dim=-1))

                    # q = policy(s_flat, lat.unsqueeze(0))
                    action = torch.argmax(q, dim=1).item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            next_state = pre.step(next_obs)

            # next state tensors

            s2_raw = next_state.unsqueeze(0).to(DEVICE)
            s2_flat = s2_raw.view(1, -1, 84, 84)

            with torch.no_grad():
                lat2 = jepa(s2_raw).squeeze(0)

            buffer.push(
                state.cpu(),
                action,
                reward,
                next_state.cpu(),
                done,
                lat.cpu(),
                lat2.cpu(),
            )

            state = next_state
            total += reward

            loss = train_step(policy, target, buffer, optim)

            if step % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            step += 1

        """
            while not done:
            eps = get_epsilon(step)

            s_t = state.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
            s_t = s_t.view(1, -1, 84, 84)  # (1, T*C, H, W)

            # s_t = state.unsqueeze(0).to(DEVICE)
            # lat = jepa(s_t).squeeze(0)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy(s_t, lat.unsqueeze(0))
                    action = torch.argmax(q, dim=1).item()

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            next_state = pre.step(next_obs)

            s2_t = next_state.unsqueeze(0).to(DEVICE)
            lat2 = jepa(s2_t).squeeze(0)

            buffer.push(state, action, reward, next_state, done, lat, lat2)

            state = next_state
            total += reward

            loss = train_step(policy, target, buffer, optim)

            if step % TARGET_UPDATE == 0:
                target.load_state_dict(policy.state_dict())

            step += 1"""
        episode += 1
        print("Episode reward:", total)
        print(f"Episode {episode} | Reward: {total:.2f} | Epsilon: {eps:.3f}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train()
