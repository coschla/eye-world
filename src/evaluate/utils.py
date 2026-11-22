import statistics

# Full ALE action space; index == raw ALE action ID, identical for every game.
ALE_ACTION_NAMES = (
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
)


def print_summary_report(
    episode_rewards,
    total_action_counts,
    mean_steps: float,
    forced_fires: int = 0,
    capped: int = 0,
    max_steps: int = 0,
) -> None:
    """The single evaluation report: rewards, then which actions the policy pressed."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Episodes     : {len(episode_rewards)}")
    print(f"Rewards      : {episode_rewards}")
    print(
        f"Mean reward  : {statistics.mean(episode_rewards):.2f} "
        f"(std {statistics.pstdev(episode_rewards):.2f}, "
        f"min {min(episode_rewards):.0f}, max {max(episode_rewards):.0f})"
    )
    print(f"Mean steps   : {mean_steps:.0f}")

    if capped:
        print(f"{capped} episode(s) stopped at gym_max_steps={max_steps}")

    if forced_fires:
        print(f"Auto-FIRE presses (not counted below): {forced_fires}")

    total = sum(total_action_counts.values())
    print(f"Actions ({total} model decisions):")
    for action_id, name in enumerate(ALE_ACTION_NAMES):
        count = total_action_counts[action_id]
        if count:
            print(f"  {action_id:2d} {name:14s} {count:7d} ({100.0 * count / total:5.1f}%)")
