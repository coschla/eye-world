import webdataset as wds
import yaml

from utils import get_train_test_files

config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
game = config["games"][0]

train_files, _ = get_train_test_files(game, config)

dataset = wds.WebDataset(
    [train_files[0]],
    shardshuffle=False,
    empty_check=False,
).to_tuple("__key__", "cls")

print("First 15 samples in the archive:\n")

for i, (key, action) in enumerate(dataset):
    if i >= 15:
        break

    if isinstance(action, bytes):
        action = action.decode()

    print(f"Position: {i:02d} | Frame key: {key} | Action: {action}")
