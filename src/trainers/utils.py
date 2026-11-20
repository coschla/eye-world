from models.jepa import encoder_frames


def format_batch_for_jepa(batch, config):
    stacked_imgs, stacked_gaze, stacked_actions = batch

    T = stacked_actions.shape[1]  # sequence length

    # [B, C*T, H, W] -> [B, T, H, W]; RGB is converted to luminance, the same as the classifier and pretraining.
    imgs = encoder_frames(stacked_imgs, T).squeeze(2)  # [B, T, H, W]

    # -----------------------------
    # Actions (already correct shape)
    # -----------------------------
    actions = stacked_actions  # [B, T]

    return imgs, actions, stacked_gaze


import torch


def atari_to_gym(actions):
    """
    Convert full ALE action IDs (0-17) to the
    6-action Space Invaders Gym/ALE action space.

    Gym SpaceInvaders:
        0 = NOOP
        1 = FIRE
        2 = RIGHT
        3 = LEFT
        4 = RIGHTFIRE
        5 = LEFTFIRE
    """

    if torch.is_tensor(actions):
        actions = actions.long()

        # Default unsupported actions to NOOP
        gym_actions = torch.zeros_like(actions, dtype=torch.long)

        gym_actions[actions == 0] = 0  # NOOP
        gym_actions[actions == 1] = 1  # FIRE

        gym_actions[actions == 3] = 2  # RIGHT
        gym_actions[actions == 4] = 3  # LEFT

        gym_actions[actions == 11] = 4  # RIGHTFIRE
        gym_actions[actions == 12] = 5  # LEFTFIRE

        return gym_actions

    mapping = {
        0: 0,  # NOOP
        1: 1,  # FIRE
        3: 2,  # RIGHT
        4: 3,  # LEFT
        11: 4,  # RIGHTFIRE
        12: 5,  # LEFTFIRE
    }

    return mapping.get(actions, 0)
