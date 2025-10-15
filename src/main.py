import yaml

from data.data_write import eye_gaze_to_webdataset
from dataset.pre_process import ComposePreprocessor, ResizePreprocessor
from dataset.torch_dataset import get_torch_dataloaders
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "data_cleaning") as check, check():
    for game in config["games"]:
        eye_gaze_to_webdataset(game, config)


with skip_run("skip", "torch_dataset") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([ResizePreprocessor(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )

    for x, y in train_test_dataloaders["train"]:
        print(x.shape)
        print(y.shape)


with skip_run("skip", "gaze_prediction") as check, check():
    game = config["games"][0]
    # TODO: Add logic here to run the training of gaze prediction
