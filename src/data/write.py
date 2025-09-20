import bz2
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import webdataset
from PIL import Image


def read_bz2_file(file_path: Path):
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


def extract_images_and_write_to_webdataset(
    tar_bz2_file: Path, writer: webdataset.TarWriter, eye_gaze
) -> None:
    """
    Decompresses the .tar.bz2 file, extracts images, and writes them to a WebDataset tar file.
    """
    decompressed_data = read_bz2_file(tar_bz2_file)
    if decompressed_data is None:
        return

    tar_bytes = BytesIO(decompressed_data)

    try:
        with tarfile.open(fileobj=tar_bytes, mode="r:") as tar:
            for num, member in enumerate(tar.getmembers()):
                if member.isfile():
                    file_data = tar.extractfile(member).read()
                    try:
                        img = Image.open(BytesIO(file_data))
                        print(eye_gaze)
                        sample = {"__key__": str(num), "image": img, "coords": eye_gaze}
                        # TODO: Add your webdataset logic here. You should save the image and the eye-gaze info
                        # sample = None
                        writer.write(sample)
                    except Exception as e:
                        print(f"Failed to open image {member.name}: {e}")
    except Exception as e:
        print(f"Error processing tar.bz2 file: {e}")


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
    For each subject and trial in the specified game, reads the corresponding eye gaze data file
    and writes images to a WebDataset tar file.
    """
    raw_data_path = Path(config["raw_data_path"]) / game
    game_meta_data = get_game_meta_data(game, config)
    output_file = Path(config["procesed_data_path"]) / game

    with webdataset.TarWriter(output_file) as writer:
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
                    # NOTE: we have not used eye-data yet, we need to write it along with images.
                    eye_gaze = pd.read_csv(file_path, sep="\t")

                    # Write the game frames to .tar files
                    tar_bz2_file = read_path.with_name(f"{read_path.name}.tar.bz2")
                    extract_images_and_write_to_webdataset(
                        tar_bz2_file, writer, f"{subject_id}_{trial_id}", eye_gaze
                    )

                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
