import pytorch_lightning as pl
import torch
from torch import nn

# ================================================================
# Lightning training module
# ================================================================


class ActionHistoryTraining(pl.LightningModule):
    """
    Trains the frames-only JEPA action classifier.

    Expected action packet:

        actions:
            [B, T]
        or:
            [B, T, 1]

    No action history is used: the network sees only the frames, and the
    target is the last action of the packet:

        target = actions[:, -1]
    """

    def __init__(
        self,
        hparams: dict,
        net: nn.Module,
        data_loader: dict,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.net = net

        self.data_loaders = data_loader

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.learning_rate = float(
            hparams.get(
                "learning_rate",
                1e-3,
            )
        )

        self.weight_decay = float(
            hparams.get(
                "weight_decay",
                0.0,
            )
        )

        self.encoder_lr = float(hparams.get("encoder_lr", 1e-5))

        self.num_actions = int(hparams["num_actions"])

        self._confusion = {}

        self._extra = {}

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(images)

    # =========================================================
    # Shared train / validation / test logic
    # =========================================================

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Existing dataset format
        #
        # images, gaze, actions
        # -----------------------------------------------------

        images, _, actions = batch

        images = images.float()

        # -----------------------------------------------------
        # Preserve compatibility with current dataset shape
        #
        # Example:
        #
        # [B,12,1,84,84]
        #
        # ->
        #
        # [B,12,84,84]
        # -----------------------------------------------------

        if images.ndim == 5:
            if images.shape[2] == 1:
                images = images.squeeze(2)

        if images.ndim != 4:
            raise ValueError(
                f"Expected classifier images [B,T*C,H,W], got {tuple(images.shape)}"
            )

        # -----------------------------------------------------
        # Actions
        # -----------------------------------------------------

        actions = torch.as_tensor(
            actions,
            device=images.device,
        )

        # Example:
        #
        # [B,T,1]
        #
        # ->
        #
        # [B,T]

        if actions.ndim >= 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        if actions.ndim != 2:
            raise ValueError(f"Expected actions [B,T], got {tuple(actions.shape)}")

        # -----------------------------------------------------
        # Current action target
        #
        # [B,T]
        #
        # ->
        #
        # [B]
        # -----------------------------------------------------

        targets = actions[
            :,
            -1,
        ].long()

        # -----------------------------------------------------
        # Debug first step
        # -----------------------------------------------------

        if self.global_step == 0 and stage == "train":
            print(
                "IMAGE SHAPE:",
                images.shape,
            )

            print(
                "FULL ACTION SHAPE:",
                actions.shape,
            )

            print(
                "TARGET SHAPE:",
                targets.shape,
            )

            print(
                "TARGET EXAMPLE:",
                targets[0],
            )

            print(
                "UNIQUE TARGET ACTIONS:",
                torch.unique(
                    targets,
                    return_counts=True,
                ),
            )

        # -----------------------------------------------------
        # Network
        #
        # images:
        # [B,12,84,84]
        #
        # ->
        #
        # logits:
        # [B,1,18]
        # -----------------------------------------------------

        logits = self.net(images)

        if logits.ndim != 3:
            raise RuntimeError(
                "Expected ActionNet output "
                "[B,1,num_actions], "
                f"got {tuple(logits.shape)}"
            )

        # -----------------------------------------------------
        # One prediction per packet
        #
        # [B,1,18]
        #
        # ->
        #
        # [B,18]
        # -----------------------------------------------------

        logits = logits[
            :,
            -1,
            :,
        ]

        # -----------------------------------------------------
        # Classification loss
        # -----------------------------------------------------

        loss = self.criterion(
            logits,
            targets,
        )

        # -----------------------------------------------------
        # Accuracy
        # -----------------------------------------------------

        predicted_actions = logits.argmax(dim=-1)

        accuracy = (predicted_actions == targets).float().mean()

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if stage != "train":
            confusion = self._confusion.setdefault(
                stage, torch.zeros(self.num_actions, self.num_actions)
            )
            index = targets * self.num_actions + predicted_actions
            confusion += (
                torch.bincount(index, minlength=confusion.numel())
                .view_as(confusion)
                .cpu()
                .float()
            )

            # The previous action is used for this metric only, never as model input.
            last = actions[:, -2].long()
            changed = targets != last
            self._extra[stage] = self._extra.get(stage, 0) + torch.stack(
                [
                    torch.tensor(len(targets), device=targets.device),
                    changed.sum(),
                    ((predicted_actions == targets) & changed).sum(),
                    (last == targets).sum(),
                ]
            ).float().cpu()

        return loss

    def _report(self, stage: str) -> None:
        """Per-class recall plus the "always predict the most common action" baseline."""
        confusion = self._confusion.pop(stage, None)

        if confusion is None or confusion.sum() == 0:
            return

        support = confusion.sum(dim=1)
        recall = confusion.diag() / support.clamp(min=1)
        present = support > 0
        balanced = recall[present].mean().item()
        majority = (support.max() / support.sum()).item()
        accuracy = (confusion.diag().sum() / support.sum()).item()

        n, changed, changed_correct, repeat = self._extra.pop(stage)
        change_acc = (changed_correct / changed.clamp(min=1)).item()
        repeat_acc = (repeat / n).item()

        self.log(f"{stage}_balanced_accuracy", balanced)
        self.log(f"{stage}_majority_baseline", majority)
        self.log(f"{stage}_change_accuracy", change_acc)
        self.log(f"{stage}_repeat_last_baseline", repeat_acc)

        if stage == "test":
            print(f"\nTEST accuracy {accuracy:.3f} | balanced {balanced:.3f} | "
                  f"majority-class baseline {majority:.3f} | "
                  f"repeat-last baseline {repeat_acc:.3f}")
            print(f"accuracy where the action changes ({int(changed)} of {int(n)} "
                  f"samples): {change_acc:.3f}")
            for action in torch.nonzero(present).flatten().tolist():
                print(f"  action {action:2d}: recall {recall[action]:.3f} "
                      f"(n={int(support[action])})")

    def on_validation_epoch_end(self):
        self._report("val")

    def on_test_epoch_end(self):
        self._report("test")

    # =========================================================
    # Lightning steps
    # =========================================================

    def training_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "train",
        )

    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "val",
        )

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "test",
        )

    # =========================================================
    # Optimizer
    # =========================================================

    def configure_optimizers(
        self,
    ):

        # Only include trainable parameters.
        #
        # If tubelet_embed and student are frozen,
        # this trains:
        #
        #   action_head

        encoder_ids = {
            id(p)
            for module in (self.net.tubelet_embed, self.net.student)
            for p in module.parameters()
        }
        encoder, rest = [], []

        for parameter in self.net.parameters():
            if parameter.requires_grad:
                (encoder if id(parameter) in encoder_ids else rest).append(parameter)

        groups = [{"params": rest, "lr": self.learning_rate}]

        if encoder:
            groups.append({"params": encoder, "lr": self.encoder_lr})

        return torch.optim.Adam(groups, weight_decay=self.weight_decay)

    # =========================================================
    # Data loaders
    # =========================================================

    def train_dataloader(
        self,
    ):

        return self.data_loaders["train"]

    def val_dataloader(
        self,
    ):

        loader = self.data_loaders.get("val")

        if loader is None:
            loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader

    def test_dataloader(
        self,
    ):

        loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader
