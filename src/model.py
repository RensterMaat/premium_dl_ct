import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks import nets
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import Accuracy, BinaryAUROC
from vit_pytorch.vit_3d import ViT
from config import lesion_level_labels_csv, dmtr_csv


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.setup_model()

        self.train_auc = BinaryAUROC(pos_label=1)
        self.val_auc = BinaryAUROC(pos_label=1)
        self.train_patient_auc = BinaryAUROC(pos_label=1)
        self.val_patient_auc = BinaryAUROC(pos_label=1)

        self.patient_labels = pd.read_csv(dmtr_csv).set_index("id")[
            config.patient_target
        ]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate_max,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.momentum > 0,
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

        loss = nn.BCELoss()(y_hat.squeeze(-1), y.float())
        self.train_auc.update(y_hat.squeeze(), y.int())

        self.log_dict(
            {
                "train_loss": loss,
                "train_auc": self.train_auc.compute(),
                "lr": self.optimizer.param_groups[0]["lr"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "preds": y_hat, "patient": batch["patient"]}

    def training_epoch_end(self, outputs):
        preds = torch.concat([output["preds"] for output in outputs])
        patient = [el for sl in outputs for el in sl["patient"]]

        patient_level_preds = self.get_patient_level_preds(preds, patient)
        patient_level_labels = self.get_corresponding_patient_level_labels(
            patient_level_preds.index
        )

        self.train_patient_auc.update(
            torch.tensor(patient_level_preds.values), patient_level_labels
        )

        self.log_dict(
            {"train_patient_auc": self.train_patient_auc.compute()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["label"]

        y_hat = torch.sigmoid(self.model(x))

        non_nan_indices = [not torch.isnan(lesion_label) for lesion_label in y]

        loss = nn.BCELoss()(
            y_hat.squeeze(-1)[non_nan_indices], y.float()[non_nan_indices]
        )
        self.val_auc.update(
            y_hat.squeeze(-1)[non_nan_indices], y.int()[non_nan_indices]
        )

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_auc": self.val_auc.compute(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "preds": y_hat, "patient": batch["patient"]}

    def validation_epoch_end(self, outputs):
        preds = torch.concat([output["preds"] for output in outputs])
        patient = [el for sl in outputs for el in sl["patient"]]

        patient_level_preds = self.get_patient_level_preds(preds, patient)
        patient_level_labels = self.get_corresponding_patient_level_labels(
            patient_level_preds.index
        )

        self.val_patient_auc.update(
            torch.tensor(patient_level_preds.values), patient_level_labels
        )

        self.log_dict(
            {"valid_patient_auc": self.val_patient_auc.compute()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def setup_model(self):
        if self.config.model == "densenet121":
            architecture = nets.densenet.DenseNet121
        elif self.config.model == "densenet169":
            architecture = nets.densenet.DenseNet169
        elif self.config.model == "densenet201":
            architecture = nets.densenet.DenseNet201
        elif self.config.model == "SEResNet50":
            architecture = nets.SEResNet50
        elif self.config.model == "SEResNet101":
            architecture = nets.SEResNet101
        elif self.config.model == "SEResNet152":
            architecture = nets.SEResNet152
        elif self.config.model == "SEResNext50":
            architecture = nets.SEResNext50
        elif self.config.model == "SEResNext101":
            architecture = nets.SEResNext101
        elif self.config.model.startswith("efficientnet"):
            self.model = nets.efficientnet.EfficientNetBN(
                self.config.model,
                spatial_dims=self.config.dim,
                in_channels=3 if self.config.dim == 2 else 1,
                num_classes=1,
                pretrained=self.config.pretrained,
            )
        elif self.config.model.startswith("vit"):
            size = self.config.model.split("_")[1]  # b or l
            patch_size = int(self.config.model.split("_")[2])  # 16 or 32

            if size == "b":
                depth = 12
                heads = 12
                dim = 768
                mlp_dim = 3072
            elif size == "l":
                depth = 24
                heads = 16
                dim = 1024
                mlp_dim = 4096

            self.model = ViT(
                image_size=128,
                frames=128,
                image_patch_size=patch_size,
                frame_patch_size=patch_size,
                num_classes=1,
                channels=1,
                depth=depth,
                heads=heads,
                dim=dim,
                mlp_dim=mlp_dim,
                dropout=self.config.dropout,
            )

        if self.config.model.startswith("densenet"):
            self.model = architecture(
                spatial_dims=self.config.dim,
                in_channels=3 if self.config.dim == 2 else 1,
                out_channels=1,
                dropout_prob=self.config.dropout,
                pretrained=self.config.pretrained,
            )
        elif self.config.model.startswith("SEResNe"):
            self.model = architecture(
                spatial_dims=self.config.dim,
                in_channels=3 if self.config.dim == 2 else 1,
                num_classes=1,
                dropout_prob=self.config.dropout,
                pretrained=self.config.pretrained,
            )

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
