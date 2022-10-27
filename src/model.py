import torch
import torch.nn as nn
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything
from monai.networks.nets.densenet import DenseNet121


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        y_hat = torch.sigmoid(self.model(x))

        loss = nn.BCELoss()(y_hat.squeeze(), y.float())

        self.log_dict(
            {"train_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        y_hat = torch.sigmoid(self.model(x))

        loss = nn.BCELoss()(y_hat.squeeze(), y.float())

        self.log_dict(
            {"valid_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss
