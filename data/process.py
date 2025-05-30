from pathlib import Path

from .read import read_gaze_data
from .utils import sort_files_by_timestamp


def process_gaze_data(game, config):
    # Paths
    raw_data_dir = config["raw_data_path"]
    read_path = Path(raw_data_dir) / game

    if read_path.exists():
        txt_files = sort_files_by_timestamp(read_path)

        for txt_file in txt_files:
            # TODO: Impelment the logic to access the images in .tar file
            read_gaze_data(txt_file)
