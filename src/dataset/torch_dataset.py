import webdataset as wds
from torch.utils.data import DataLoader

from .utils import get_train_test_files, get_train_val_test_files


def read_webdataset(file_list, config):
    dataset = (
        wds.WebDataset(
            file_list,
            shardshuffle=False,
            nodesplitter=wds.split_by_worker,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple("jpg", "json", "cls")
    )
    return dataset


def create_dataloader(file_list, config, preprocessor=None, shuffle=False):
    dataset = read_webdataset(file_list, config)

    # Apply optional preprocessor (frame stacking needs the original order)
    if preprocessor:
        dataset = dataset.map(preprocessor)

    # Iterable dataset: shuffle samples in a buffer, not through the DataLoader
    if shuffle:
        dataset = dataset.shuffle(int(config["shuffle_buffer"]))

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )


"""def get_torch_dataloaders(game, config, preprocessor=None):
    train_files, test_files = get_train_test_files(game, config)
    # the files are named in order but there in nothing that takes advange of that the files list are only the name of the test files
    return {
        "train": create_dataloader(train_files, config, preprocessor),
        "test": create_dataloader(test_files, config, preprocessor),
    }
"""


def get_torch_dataloaders(game, config, preprocessor=None, use_val=False):
    """use_val=True holds config["val_sub"] out of train and adds a "val" loader."""
    games = game if isinstance(game, (list, tuple, set)) else [game]
    def split(g, c):
        if use_val:
            return get_train_val_test_files(g, c)
        train, test = get_train_test_files(g, c)
        return train, [], test

    train, val, test = [], [], []
    for g in games:
        t, v, e = split(g, config)
        train += t
        val += v
        test += e
    loaders = {
        "train": create_dataloader(train, config, preprocessor, config["shuffle"]),
        "test": create_dataloader(test, config, preprocessor),
    }
    if use_val and val:
        loaders["val"] = create_dataloader(val, config, preprocessor)
    return loaders
