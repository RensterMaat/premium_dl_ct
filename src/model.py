import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
from torchmetrics.classification import BinaryAUROC, Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything
from monai.networks.nets.densenet import DenseNet121


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        self.aggregation_function = config.aggregation_function

        self.lr = config.lr
        self.t_max = config.t_max
        self.lr_min = config.lr_min

        self.train_auc = BinaryAUROC(pos_label=1)
        self.val_auc = BinaryAUROC(pos_label=1)

        self.train_patient_auc = BinaryAUROC(pos_label=1)
        self.val_patient_auc = BinaryAUROC(pos_label=1)

        self.patient_labels = (
            pd.read_csv(r"C:\Users\user\data\tables\dmtr.csv")
            .set_index("id")["locafmetlong"]
            .fillna(0)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.t_max, eta_min=self.lr_min
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]
        y_hat = torch.sigmoid(self.model(x))

        loss = nn.BCELoss()(y_hat.squeeze(), y.float())
        self.train_auc.update(y_hat.squeeze(), y.int())

        patient_level_preds = self.get_patient_level_preds(y_hat, batch["patient"])
        patient_level_labels = self.get_corresponding_patient_level_labels(
            patient_level_preds.index
        )

        self.train_patient_auc.update(
            torch.tensor(patient_level_preds.values), patient_level_labels
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_auc": self.train_auc.compute(),
                "train_patient_auc": self.train_patient_auc.compute(),
            },
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
        self.val_auc.update(y_hat.squeeze(), y.int())

        patient_level_preds = self.get_patient_level_preds(y_hat, batch["patient"])
        patient_level_labels = self.get_corresponding_patient_level_labels(
            patient_level_preds.index
        )

        self.val_patient_auc.update(
            torch.tensor(patient_level_preds.values), patient_level_labels
        )

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_auc": self.val_auc.compute(),
                "valid_patient_auc": self.val_patient_auc.compute(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def get_patient_level_preds(self, preds, patients):
        results = pd.DataFrame(
            [patients, preds.squeeze().detach().cpu()], index=["patient", "preds"]
        ).transpose()
        patient_level_preds = (
            results.groupby("patient")
            .preds.apply(self.get_aggregation_function())
            .apply(np.array)
        )

        return patient_level_preds

    def get_corresponding_patient_level_labels(self, patients):
        return torch.tensor(self.patient_labels.loc[patients].values)

    def get_aggregation_function(self):
        if self.aggregation_function == "mean":
            return np.mean
        elif self.aggregation_function == "max":
            return np.max
        elif self.aggregation_function == "min":
            return np.min
