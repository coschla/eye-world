import pytorch_lightning as pl
import torch
import torch.nn as nn


class ActionTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()

        self.model = net
        self.data_loader = data_loader

        # classification loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        stacked_imgs, stacked_gaze, stacked_actions = batch

        # reshape batch like in format_batch_for_vjepa
        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]  # sequence length
        C = CT // T  # channels per frame

        # reshape: [B, C*T, H, W] → [B, T, C, H, W]
        x = stacked_imgs.view(B, T, C, H, W)

        # optionally convert grayscale
        if C == 1:
            x = x.squeeze(2)  # → [B, T, H, W]
        else:
            x = x.mean(dim=2)  # average over channels → [B, T, H, W]

        # last action as target
        y = stacked_actions[:, -1]

        if batch_idx == 0:
            print("\n===== ACTION DEBUG (OLD VERSION) =====")
            print("stacked_actions shape:", stacked_actions.shape)
            print("sample actions:", stacked_actions[0])
            print("unique actions:", torch.unique(stacked_actions))
            print("y:", y[:10])
            print("y unique:", y.unique())
            print("=====================================\n")

        logits = self(x)
        loss = self.criterion(logits, y)

        # 🔹 Debug prints
        if batch_idx == 0:  # only first batch
            print("===== DEBUG: training_step =====")
            print("stacked_imgs.shape:", stacked_imgs.shape)
            print("stacked_actions.shape:", stacked_actions.shape)
            print("x.shape (input to net):", x.shape)
            print("y.shape (target):", y.shape)
            print("y min/max:", y.min().item(), y.max().item())
            print("y unique values:", y.unique())
            print("dtype:", y.dtype)
            print("num_actions (model output):", self.model.fc_layers[-1].out_features)
            print("===============================")

        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        print("===== RAW BATCH DEBUG =====")
        print("type(batch):", type(batch))

        if isinstance(batch, (list, tuple)):
            print("len(batch):", len(batch))
            for i, item in enumerate(batch):
                if hasattr(item, "shape"):
                    print(f"item {i} shape:", item.shape)
                    print(f"item {i} dtype:", item.dtype)
                else:
                    print(f"item {i} type:", type(item))
            print("===========================")
        stacked_imgs, stacked_gaze, stacked_actions = batch

        # reshape batch
        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]
        C = CT // T

        x = stacked_imgs.view(B, T, C, H, W)
        if C == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)

        y = stacked_actions[:, -1]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        # self.log("val_acc", acc, on_epoch=True, on_step=False, sync_dist=True)

        # 👇 map loader → game name

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            "val_acc_global",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        return loss

    def val_dataloader(self):
        return self.data_loader["test"]  # single mixed loader OR single dataset

    def train_dataloader(self):
        return self.data_loader["train"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2.5e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


"""


class ActionTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()

        self.model = net
        self.data_loader = data_loader
        self.num_future = net.num_future

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        stacked_imgs, stacked_gaze, stacked_actions = batch

        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]
        C = CT // T

        x = stacked_imgs.reshape(B, T, C, H, W)

        if C == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)

        # =========================
        # 🔍 DEBUG: ACTION ALIGNMENT CHECK
        # =========================
        if batch_idx == 0:
            print("\n===== FUTURE ACTION DEBUG =====")

            print("stacked_actions shape:", stacked_actions.shape)
            print("num_future:", self.num_future)

            # show full sequence for first sample
            print("\nSample action sequence [0]:")
            print(stacked_actions[0].tolist())

            y = stacked_actions[:, -self.num_future :]

            print("\nTarget y shape:", y.shape)
            print("Target y[0]:", y[0].tolist())

            print("\n🔎 Checking alignment assumption:")
            print("Last action in sequence:", stacked_actions[0, -1].item())
            print("Second last:", stacked_actions[0, -2].item())

            print("\n⚠️ If this is NOT t+1, t+2, your dataset is not future-aligned.")
            print("================================\n")

            if batch_idx == 0:
                print("FULL ACTION ROWS:")
                print(stacked_actions[:5])

        # =========================
        # TARGET
        # =========================
        y = stacked_actions[:, -self.num_future :]  # supposed future actions

        logits = self(x)

        B, T, A = logits.shape

        logits_flat = logits.reshape(B * T, A)
        y_flat = y.reshape(B * T)

        # =========================
        # 🔍 DEBUG: FINAL SANITY CHECK
        # =========================
        if batch_idx == 0:
            print("logits shape:", logits.shape)
            print("logits_flat shape:", logits_flat.shape)
            print("y_flat shape:", y_flat.shape)

            print("\nAction range check:")
            print("y min:", y_flat.min().item())
            print("y max:", y_flat.max().item())
            print("num_actions (expected upper bound):", A)

            assert y_flat.min() >= 0, "Negative action found!"
            assert y_flat.max() < A, "Action index exceeds num_actions!"

        loss = self.criterion(logits_flat, y_flat)

        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        stacked_imgs, stacked_gaze, stacked_actions = batch

        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]
        C = CT // T

        x = stacked_imgs.view(B, T, C, H, W)

        if C == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)

        y = stacked_actions[:, -self.num_future :]  # [B, 2]

        logits = self(x)  # [B, 2, A]

        B, T, A = logits.shape
        logits_flat = logits.reshape(B * T, A)
        y_flat = y.reshape(B * T)

        loss = self.criterion(logits_flat, y_flat)

        preds = torch.argmax(logits_flat, dim=1)
        acc = (preds == y_flat).float().mean()

        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, sync_dist=True)

        return loss

    def train_dataloader(self):
        return self.data_loader["train"]

    def val_dataloader(self):
        return self.data_loader["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
"""
