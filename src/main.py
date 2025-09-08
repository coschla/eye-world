import yaml

from data.write import eye_gaze_to_webdataset
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("run", "data_cleaning") as check, check():
    games = config["games"]
    eye_gaze_to_webdataset(games[0], config)
