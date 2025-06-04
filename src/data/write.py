import webdataset
import json
import io
from .read import read_game_frames, read_gaze_data


def write_eye_gaze_dataset(file_path):
    with webdataset.TarWriter("dataset__pred_1.tar") as writer:
        coord, img = read_gaze_data(file_path), read_game_frames(file_path)

        sample = {"__key__": str(x), "img": img, "coord": coord}

        writer.write(sample)
