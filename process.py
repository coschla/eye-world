# from read import read_gaze_data
# from utils import sort_files_by_timestamp


import webdataset as wds


def process_gaze_data(config_path, key_number):
    key_str = str(key_number)

    dataset = (
        wds.WebDataset(config_path, shardshuffle=False)
        .decode("pil")  # decode .png into PIL images
        .to_tuple("__key__", "png", "json")  # load sample as (key, image, json)
    )

    for key, img, coords in dataset:
        if key == key_str:
            return [img], [coords]  # wrap in list to keep return structure

    return [], []  # fallback if not found


# Example usage
images, coords = process_gaze_data(
    "file:C:/Users/X570 MASTER/Desktop/redue/eye-world/dataset__pred_1.tar", 500
)
print(coords)
if images:
    images[0].show()  # Show the first image
else:
    print("No image found.")
