import matplotlib.pyplot as plt
import pytorch_lightning as pl
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

from data.data_write import eye_gaze_to_webdataset
from dataset.pre_process import ComposePreprocessor, ResizePreprocessor
from dataset.torch_dataset import get_torch_dataloaders
from models.networks import ConvNet
from trainers.gaze_predict import GazeTraining
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
plt.ion()

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

with skip_run("run", "gaze_prediction") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([ResizePreprocessor(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    print(train_test_dataloaders)

    fig, ax = plt.subplots()
    for batch_idx, (imgs, labels) in enumerate(train_test_dataloaders["train"]):
        for i in range(len(imgs)):
            img = imgs[i]
            label = labels[i]
            img_np = img.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.set_title(f"Frame {batch_idx},{i} Lable : {label}")
            plt.pause(0.5)
            ax.clear()
    plt.ioff
    plt.show()


with skip_run("skip", "gaze_prediction") as check, check():
    logger = TensorBoardLogger("tb_logs", name="test_light")

    game = config["games"][0]
    # gaze prediction network
    net = ConvNet(
        config=config,
    )

    # Dataloader
    preprocessor = ComposePreprocessor([ResizePreprocessor(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    model = GazeTraining(config, net, train_test_dataloaders)

    # Trainer
    trainer = pl.Trainer(
<<<<<<< HEAD
        max_epochs=1,
=======
        # gpus=1,
        max_epochs=100,
>>>>>>> dd199f7 (frame visulization created)
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model)
