import yaml

# from eye_gaze_pred import eye_gaze_
# from data.process import process_gaze_data
# from utils import skip_run
from data_L import data_load_

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

train_data, test_data = data_load_(
    folder=config["training_data_path"], batch=1, config=config
)

# train_size = len(train_data)
# test_size = len(test_data)
print(next(iter(test_data)))
# eye_gaze_(config)
print("a")

"""
with skip_run("run", "data_cleaning") as check, check():
    games = config["games"]

    for game, valid_actions in games.items():
        process_gaze_data(game, config)
"""
