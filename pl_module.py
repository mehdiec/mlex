import warnings as warn 
import pytorch_lightning as pl
import torch
from torch import nn 

class ClassificationPLModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=2e-5,
        epoch=100,
        weight_decay=0.001,
        scheduler="cosine",
        pos_weight=0.2,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        weight_tensor = torch.tensor(pos_weight, device=next(self.model.parameters()).device) 
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        self.train_outputs = []
        self.train_labels = []
        self.val_outputs = []
        self.val_labels = []

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]

        outputs = self(inputs)
        loss = self.loss(outputs.flatten(), labels.float())
        outputs = torch.sigmoid(outputs)

        # Store outputs and labels for metrics calculation
        self.train_outputs.extend(outputs.flatten().detach().cpu().numpy())
        self.train_labels.extend(labels.cpu().numpy())

        # Log loss and learning rate
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]

        outputs = self(inputs)
        loss = self.loss(outputs.flatten(), labels.float())
        outputs = torch.sigmoid(outputs)

        # Store outputs and labels for metrics calculation
        self.val_outputs.extend(outputs.flatten().detach().cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())

        # Log validation loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0], batch[1]
        outputs = self(inputs)
        loss = self.loss(outputs.flatten(), labels.float())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler == "one_cycle":
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            )
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=2e-3,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.15,
                three_phase=False,
            )
            scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",  # Changed from 'epoch' to 'step'
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
            }
            return [optimizer], [scheduler_config]
        elif self.scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
            )
        elif self.scheduler == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=min(10, self.trainer.max_epochs // 10),
                factor=0.1,
            )
        else:
            warn.warn(
                f"scheduler {self.scheduler} not recognised running ReduceLROnPlateau instead"
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=min(10, self.trainer.max_epochs // 10),
                factor=0.1,
            )

        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
        }

        return [optimizer], [scheduler_config]
