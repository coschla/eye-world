import os
import tarfile

from torch.utils.data import DataLoader

# make into a function


def data_load_(folder, batch, config):
    train_data = []
    val_data = []
    val_sub = config["validation_sub"]
    train_sub = config["train_sub"]

    for player in os.listdir(folder):
        player_path = folder + "/" + player
        for fname in os.listdir(player_path):
            name = folder + "/" + player + "/" + fname
            print(folder)
            with tarfile.open(name, "r") as tar:
                members = tar.getmembers()
                length = len([m for m in members if m.isfile()])
                print(player)
                print(fname)
                player_part = player.split("_")[1]
                trial_part = fname.split("_")[1]
                trial_num = trial_part.split(".")[0]
                sub = player_part + "_" + trial_num
                print(sub)
                if (sub) in val_sub:
                    for num in range(length // 2):
                        address = sub + "_" + str(num)

                        val_data.append(address)
                if (sub) in train_sub:
                    for num in range(length // 2):
                        address = sub + "_" + str(num)
                        train_data.append(address)

    train_data_loader = DataLoader(train_data, batch_size=batch)
    validation_data_loader = DataLoader(val_data, batch_size=batch)
    return train_data_loader, validation_data_loader
