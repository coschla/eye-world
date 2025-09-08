import bz2
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import webdataset


def read_bz2_file(file_path: Path) -> Optional[bytes]:
    """
    Reads a .bz2 compressed file and returns its decompressed content as bytes.
    """
    try:
        with bz2.open(file_path, "rb") as bz2f:
            return bz2f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
    return None


def get_game_meta_data(game: str, config: dict) -> pd.DataFrame:
    """
    Reads meta data CSV from config and returns DataFrame grouped by subject_id with list of trial_ids for the given game.
    """
    meta_data_path = Path(config["meta_data_path"])
    meta_data = pd.read_csv(meta_data_path)
    game_meta_data = (
        meta_data[meta_data["GameName"] == game]
        .groupby("subject_id")["trial_id"]
        .apply(list)
        .reset_index()
    )
    return game_meta_data


def eye_gaze_to_webdataset(game: str, config: dict) -> None:
    """
    For each subject and trial in the specified game, reads the corresponding eye gaze data file.
    """
    raw_data_path = Path(config["raw_data_path"]) / game
    game_meta_data = get_game_meta_data(game, config)

    for _, row in game_meta_data.iterrows():
        subject_id = row["subject_id"]
        trial_ids = row["trial_id"]

        print(f"Reading data for subject: {subject_id}")

        for run, trial_id in enumerate(trial_ids):
            pattern = f"{trial_id}_{subject_id}*.txt"
            matched_files = list(raw_data_path.glob(pattern))

            if not matched_files:
                print(f"Warning: No files found for pattern '{pattern}'")
                continue

            file_path = matched_files[0]
            read_path = file_path.with_suffix("")  # remove '.txt'

            try:
                eye_gaze = pd.read_csv(file_path, sep="\t")
                print(eye_gaze)
                print(f"Run {run} - Loaded {file_path.name}")
                # Do something with eye_gaze here...
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")


def write_eye_gaze_dataset(file_path, root_path):
    for path in os.listdir(root_path):
        path = root_path + "/" + path
        coord = read_gaze_data(path)
    print(coord)

    # Method A: using .apply
    df = coord["gaze_positions"]  # .apply(lambda coords: coords[0] if coords else None)
    print(df)
    img_dir = Path("src") / "data" / "game_frames" / "Class1"

    for path in os.listdir(root_path):
        print(path)
        path = root_path + "/" + path
        split_root = path.split("/")[-1]
        name_part0 = split_root.split("_")[0]
        name_part1 = split_root.split("_")[1]
        name_part2 = split_root.split("_")[2]
    prefix = name_part1 + "_" + name_part2
    print(name_part1)
    print(name_part2)
    print(prefix)
    file_name = name_part0 + "_" + prefix + "_pred.tar"
    print(file_name)
    with webdataset.TarWriter(file_name) as writer:
        for num in range(len(os.listdir(file_path))):
            name = prefix + "_" + str(num + 1) + ".png"

            img_path = img_dir / name
            img = img_path.read_bytes()
            sample = {"__key__": str(num + 1), "png": img, "json": df[num]}

            writer.write(sample)
