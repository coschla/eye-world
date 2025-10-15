import webdataset as wds
from torch.utils.data import DataLoader

from .utils import get_train_test_files


def read_webdataset(file_list, config):
    dataset = (
        wds.WebDataset(file_list, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "json")
    )
    return dataset


def create_dataloader(file_list, config, preprocessor=None):
    dataset = read_webdataset(file_list, config)

    # Apply optional preprocessor
    if preprocessor:
        dataset = dataset.map(preprocessor)

    return DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )


def get_torch_dataloaders(game, config, preprocessor=None):
    train_files, test_files = get_train_test_files(game, config)

    return {
        "train": create_dataloader(train_files, config, preprocessor),
        "test": create_dataloader(test_files, config, preprocessor),
    }
