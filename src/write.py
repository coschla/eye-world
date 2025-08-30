import webdataset
import json
import io
from read import read_gaze_data
import os
from pathlib import Path
import io
import json


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


write_eye_gaze_dataset(
    r"src/data/game_frames/Class1",
    r"src/data/eye_gaze_root",
)
print("a")
