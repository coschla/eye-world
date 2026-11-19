"""
Offline evaluation of the action-conditioned V-JEPA world model.

For every test batch it computes three smooth-L1 losses against the teacher
latents of the target frames:

  model     predictor(context latents, TRUE action)
  shuffled  predictor(context latents, action taken from another sample)
  copy      context latents used directly as the prediction

Healthy model:  model < copy   (it learned dynamics)
                shuffled > model (it uses the action)

Gaze masking is not applied (use_gaze_for_predictor / use_gaze_for_loss must
be false), matching the current config.
"""

import torch
import torch.nn.functional as F

from trainers.utils import format_batch_for_vjepa


def load_world_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def _predict(model, student_tokens, action):
    action_token = model.action_embed(action).unsqueeze(1)
    x = torch.cat([student_tokens, action_token], dim=1)
    return model.latent_predictor(x, x)[:, :-1]


@torch.no_grad()
def evaluate_world_model(model, dataloader, config, max_batches=None):
    if config.get("use_gaze_for_predictor") or config.get("use_gaze_for_loss"):
        raise NotImplementedError("Gaze masking is not supported in this evaluator.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    context = int(config["context_frames"])

    sums = {"model": 0.0, "shuffled": 0.0, "copy": 0.0}
    n = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        batch = [b.to(device) for b in batch]
        img, actions, _ = format_batch_for_vjepa(batch, config)
        if isinstance(img, list):
            img = torch.stack(img)

        student_x = img[:, :context].unsqueeze(2)
        teacher_x = img[:, context:].unsqueeze(2)

        student_tokens = model.model.student(model.model.tubelet_embed(student_x))
        target = model.teacher(model.model.tubelet_embed(teacher_x))

        action = actions[:, context - 1].long()
        B = action.shape[0]
        # Roll by one so every sample gets another sample's action.
        shuffled = action.roll(1, dims=0)

        pred = _predict(model, student_tokens, action)
        pred_shuf = _predict(model, student_tokens, shuffled)

        sums["model"] += F.smooth_l1_loss(pred, target).item() * B
        sums["shuffled"] += F.smooth_l1_loss(pred_shuf, target).item() * B
        sums["copy"] += F.smooth_l1_loss(student_tokens, target).item() * B
        n += B

    results = {k: v / max(n, 1) for k, v in sums.items()}
    results["samples"] = n

    print("\n=== World model evaluation ===")
    print(f"samples            : {n}")
    print(f"model loss         : {results['model']:.5f}")
    print(f"copy baseline      : {results['copy']:.5f}")
    print(f"shuffled-action    : {results['shuffled']:.5f}")
    print(
        "beats copy baseline:",
        "YES" if results["model"] < results["copy"] else "NO (no dynamics learned)",
    )
    print(
        "uses actions       :",
        "YES" if results["shuffled"] > 1.05 * results["model"] else "NO / weak",
    )
    return results
