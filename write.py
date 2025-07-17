import webdataset
import json
import io
from read import  read_gaze_data
import os
from pathlib import Path
import io, json


def write_eye_gaze_dataset(file_path,root_path):
    with webdataset.TarWriter("dataset__pred_1.tar") as writer:
        coord  = read_gaze_data(root_path)
        print(coord)
        # Method A: using .apply
        df = coord["gaze_positions"].apply(lambda coords: coords[-1] if coords else None)
        
        img_dir = Path("src") / "data" / "game_frames" / "Class1"
        prefix  = "RZ_2394668"


        for num in range(len (os.listdir(file_path))):
            end = num+1
            end =str(end)
            end =end + '.png'
            name =prefix + '_' + end
            img_path = img_dir / name
            img =img_path.read_bytes()
            sample = {"__key__": str(num+1), "png": img, "json": df[num]}

            writer.write(sample)

write_eye_gaze_dataset(r"C:\Users\X570 MASTER\Desktop\redue\eye-world\src\data\game_frames\Class1",r"C:\Users\X570 MASTER\Desktop\redue\eye-world\src\data\eye_gaze_root\52_RZ_2394668_Aug-10-14-52-42.txt")
print('a')