from torch.utils.data import DataLoader

from .utils import get_train_test_files


def preprocessor(samples):
    # TODO: Add pre-processing pipeline here and return processed  samples. You should return webdataset
    # NOTE: look up how to add custom pre-processing for webdataset

    return NotImplementedError


def read_webdataset(file_list, config):
    # TODO: Add logic to read web-dataset, it is very simple. Look up how to read webdataset
    dataset = None
    return dataset


def create_dataloader(file_list, config, preprocessor=None):
    dataset = read_webdataset(file_list, config)

    # Apply optional preprocessor
    if preprocessor:
        dataset = dataset.map(preprocessor)

    return DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )


def get_torch_dataloaders(config, preprocessor=None):
    train_files, test_files = get_train_test_files(config)

    return {
        "train": create_dataloader(train_files, config, preprocessor),
        "test": create_dataloader(test_files, config, preprocessor),
    }
