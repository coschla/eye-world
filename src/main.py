import yaml

from data.process import process_gaze_data
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "data_cleaning") as check, check():
    games = config["games"]

    for game, valid_actions in games.items():
        process_gaze_data(game, config)
