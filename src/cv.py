import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from model import Model
from data import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import radiomics_folder, lesion_level_labels_csv


def train():
    wandb.init()
    wandb.config.roi_selection_method = "crop"
    # wandb.config.dim = 2
    wandb.config.size = 128 if wandb.config.dim == 3 else 256
    wandb.config.test_center = None  # "amphia"
    wandb.config.lesion_target = "lesion_response"
    wandb.config.patient_target = "response"
    wandb.config.max_batch_size = 6 #if wandb.config.dim == 3 else 32
    wandb.config.seed = 0
    wandb.config.max_epochs = 100
    wandb.config.patience = 10
    wandb.config.lr_min = 1e-7
    wandb.config.T_0 = 10

    seed_everything(wandb.config.seed)

    dm = DataModule(
        radiomics_folder,
        lesion_level_labels_csv,
        wandb.config,
    )

    model = Model(wandb.config)

    logger = WandbLogger(
        name="hello7",
        project="debugging",
    )

    checkpoint_callback = ModelCheckpoint(monitor="valid_patient_auc", mode="max")

    early_stopping = EarlyStopping(
        monitor="valid_patient_auc", mode="max", patience=wandb.config.patience
    )

    trainer = Trainer(
        max_epochs=wandb.config.max_epochs,
        gpus=1,
        # deterministic=True,
        fast_dev_run=False,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    wandb.init()
    wandb.config.aggregation_function = "min"
    wandb.config.roi_size = 150
    wandb.config.optimizer = "adamw"
    wandb.config.weight_decay = 0.00001
    wandb.config.model = "SEResNet50"
    wandb.config.dropout = 0.07292136035956572
    wandb.config.momentum = 0
    wandb.config.pretrained = False
    wandb.config.learning_rate_max = 0.000023059510738335888
    wandb.config.sampler = "stratified"
    wandb.config.dim = 2

    train()

    wandb.finish()

# def infer(model, datamodule):
#     pass


# def leave_one_center_out_cross_validate(model, datamodule):
#     pass
# def leave_one_center_out_cross_validate(model, datamodule):
#     pass
