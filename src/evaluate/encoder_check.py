"""
Does the frozen encoder produce latents that carry information about the frame?

Everything is measured on layer-normed latents n [B, W, N, D] of sliding
context windows, so the numbers are scale-free:

  between_std      std of a feature across different samples
  const_energy     share of the latent that is identical for every sample
  cos_between      cosine similarity between latents of different samples
  step_change      rms change of a latent from one window to the next
  step/between     step_change relative to between_std; near 0 means the
                   latent hardly moves, so a world model can only copy it
  effective_rank   how many feature directions are actually used (of D)
"""

import torch
import torch.nn.functional as F

from trainers.utils import format_batch_for_jepa


@torch.no_grad()
def check_encoder(encoder, dataloader, config, max_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    ctx = int(config["context_frames"])
    latents = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        img, _, _ = format_batch_for_jepa([b.to(device) for b in batch], config)
        windows = img.unfold(1, ctx, 1).permute(0, 1, 4, 2, 3)  # [B, W, ctx, H, W]
        B, W = windows.shape[:2]
        z = encoder(windows.reshape(B * W, *windows.shape[2:]))
        latents.append(F.layer_norm(z, (z.size(-1),)).view(B, W, *z.shape[1:]).float())

    n = torch.cat(latents)  # [S, W, N, D]
    first = n[:, 0]  # one window per sample: [S, N, D]
    mean = first.mean(0, keepdim=True)
    between = first.std(0).mean().item()
    const = (mean.pow(2).mean() / first.pow(2).mean()).item()
    flat = F.normalize(first.flatten(1), dim=-1)
    cos = ((flat @ flat.T).sum() - flat.size(0)) / (flat.size(0) * (flat.size(0) - 1))
    step = (n[:, 1:] - n[:, :-1]).pow(2).mean().sqrt().item()
    rows = (first - mean).reshape(-1, first.size(-1))
    sv = torch.linalg.svdvals(rows[:: max(1, rows.size(0) // 4096)])
    p = sv / sv.sum()
    rank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()

    out = {
        "samples": n.size(0),
        "between_std": between,
        "const_energy": const,
        "cos_between": cos.item(),
        "step_change": step,
        "step/between": step / max(between, 1e-12),
        "effective_rank": rank,
    }
    print("\n=== Encoder latent check ===")
    for k, v in out.items():
        print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k:15s}: {v}")
    return out
