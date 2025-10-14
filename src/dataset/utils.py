from pathlib import Path

import pandas as pd


def get_game_meta_data(game: str, config: dict) -> pd.DataFrame:
    """
    Reads meta data CSV from config and returns DataFrame grouped by subject_id with list of trial_ids for the given game.
    """
    meta_data_path = Path(config["meta_data_path"])
    meta_data = pd.read_csv(meta_data_path)
    game_meta_data = (
        meta_data[meta_data["GameName"] == game]
        .groupby("subject_id")["trial_id"]
        .apply(list)
        .reset_index()
    )
    return game_meta_data


def get_train_test_files(game, config):
    game_meta_data = get_game_meta_data(game, config)
    train = []
    test = []
    read_path = config["processed_data_path"] + f"{game}/"
    for _, row in game_meta_data.iterrows():
        subject_id = row["subject_id"]
        trial_ids = row["trial_id"]
        name = [
            read_path + subject_id + "_" + str(trial_id) + ".tar"
            for trial_id in trial_ids
        ]
        if subject_id in config["train_data_user"]:
            # name = subject_id + "_" + trial_ids + ".tar"
            train.append(name)

        if subject_id in config["test_data_user"]:
            test.append(name)

    return train[0], test[0]
