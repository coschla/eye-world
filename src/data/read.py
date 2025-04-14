from pathlib import Path

import pandas as pd


def read_gaze_data(file_path):
    """
    Reads a CSV where the first 6 columns are fixed, followed by variable-length gaze data
    stored as [x, y] pairs in a list of lists.

    Args:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with fixed columns and 'gaze_positions' as list of [x, y].
    """
    data = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                continue  # Skip header

            parts = line.strip().split(",")
            if len(parts) < 6:
                continue  # skip malformed

            # First 6 fields
            fixed = parts[:6]
            gaze_raw = parts[6:]

            # Convert gaze to list of [x, y]
            gaze_positions = [
                [float(gaze_raw[i]), float(gaze_raw[i + 1])]
                for i in range(0, len(gaze_raw) - 1, 2)
            ]

            data.append(fixed + [gaze_positions])

    columns = [
        "frame_id",
        "episode_id",
        "score",
        "duration(ms)",
        "unclipped_reward",
        "action",
        "gaze_positions",
    ]
    df = pd.DataFrame(data, columns=columns)

    # Optional type conversion
    for col in ["score", "duration(ms)", "unclipped_reward", "action"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
