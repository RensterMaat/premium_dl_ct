import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import densenet
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy, BinaryAUROC


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        if config.model == "densenet121":
            architecture = densenet.DenseNet121
        elif config.model == "densenet169":
            architecture = densenet.DenseNet169
        elif config.model == "densenet201":
            architecture = densenet.DenseNet201

        self.model = architecture(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            dropout_prob=config.dropout,
            pretrained=config.pretrained,
        )

        self.config = config

        self.train_auc = BinaryAUROC(pos_label=1)
        self.val_auc = BinaryAUROC(pos_label=1)

        self.train_patient_auc = BinaryAUROC(pos_label=1)
        self.val_patient_auc = BinaryAUROC(pos_label=1)

        # self.patient_labels = (
        #     pd.read_csv(r"C:\Users\user\data\tables\dmtr.csv")
        #     .set_index("id")["locafmetlong"]
        #     .fillna(0)
        # )
        self.patient_labels = (
            pd.read_csv(
                r"C:\Users\user\data\tables\lesion_followup_curated_v4.csv", sep=";"
            )
            .groupby("patient")
            .lung.max()
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate_max,
                weight_decay=self.config.weight_decay,
                nesterov=True,
                momentum=self.config.momentum,
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate_max,
                weight_decay=self.config.weight_decay,
            )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.config.T_0, eta_min=self.config.lr_min
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
                "lr": self.optimizer.param_groups[0]["lr"],
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
        patients = [
            pt.replace("abdomen", "").replace("thorax", "").replace("hals", "")
            for pt in patients
        ]
        return torch.tensor(self.patient_labels.loc[patients].values)

    def get_aggregation_function(self):
        if self.config.aggregation_function == "mean":
            return np.mean
        elif self.config.aggregation_function == "max":
            return np.max
        elif self.config.aggregation_function == "min":
            return np.min
