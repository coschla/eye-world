import yaml
from data.data_L import data_load_
from data.data_write import eye_gaze_to_webdataset
from eye_model import eyegaze_model
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("run", "data_cleaning") as check, check():
    games = config["games"]
    eye_gaze_to_webdataset(games[0], config)

with skip_run("run") as check, check():
    data, valid = data_load_
    model = eyegaze_model()
    logger = TensorBoardLogger("tb_logs", name="test_light")
    trainer = L.Trainer(max_epochs=100, logger=logger)
    trainer.fit(eyegaze_model, train_dataloaders=data, validation_dataloaders=valid)
