from pathlib import Path
import pandas as pd


def read_gaze_data_txt(file_path):
    """
    Reads a .txt file with the same format as your CSV,
    optionally wrapped in parentheses,
    and returns a DataFrame with columns
    [frame_id, episode_id, score, duration(ms), unclipped_reward, action, gaze_positions].
    """
    # file_path = Path(file_path)
    file_path = Path(r"C:\Users\X570 MASTER\Desktop\data\eye_gaze_root\52_RZ_2394668_Aug-10-14-52-42.txt")

    # 1) load entire text, strip leading/trailing parentheses if present
    raw = file_path.read_text(encoding="utf-8").strip().strip("()")
    lines = [ln for ln in raw.splitlines() if ln.strip()]

    # 2) header & data lines
    header = lines[0].split(",")
    data = []

    for line in lines[1:]:
        parts = line.strip().strip("()").split(",")
        if len(parts) < 7:
            continue  # malformed or too short
        fixed = parts[:6]  # the first 6 fields
        gaze_raw = parts[6:]  # everything after that

        # build list of [x,y] floats
        gaze_positions = [
            [float(gaze_raw[i]), float(gaze_raw[i + 1])]
            for i in range(0, len(gaze_raw) - 1, 2)
        ]
        data.append(fixed + [gaze_positions])

    # 3) DataFrame & numeric‐conversion
    df = pd.DataFrame(data, columns=header)
    df.replace("null", pd.NA, inplace=True)  # turn "null"→<NA>
    for col in ["score", "duration(ms)", "unclipped_reward", "action"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
