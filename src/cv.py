import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from model import Model
from data import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from config import radiomics_folder, lesion_level_labels_csv, mini_batch_size


def train():
    wandb.init()
    wandb.config.roi_selection_method = "crop"
    wandb.config.size = 182  # if wandb.config.dim == 3 else 256
    wandb.config.lesion_target = "lesion_response"
    wandb.config.patient_target = "response"
    wandb.config.max_batch_size = 6  # if wandb.config.dim == 3 else 32
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
        project="survival",
    )

    checkpoint_callback = ModelCheckpoint(monitor="valid_patient_auc", mode="max")

    early_stopping = EarlyStopping(
        monitor="valid_patient_auc", mode="max", patience=wandb.config.patience
    )

    trainer = Trainer(
        max_epochs=wandb.config.max_epochs,
        gpus=1,
        # deterministic=True,
        accumulate_grad_batches=wandb.config.n_forward_per_backwards,
        fast_dev_run=False,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    wandb.init()
    wandb.config.aggregation_function = "min"
    wandb.config.roi_size = 142
    wandb.config.optimizer = "adamw"
    wandb.config.weight_decay = 1e-7
    wandb.config.model = "SEResNet50"
    wandb.config.dropout = 0
    wandb.config.momentum = 0
    wandb.config.pretrained = False
    wandb.config.learning_rate_max = 1e-5
    wandb.config.sampler = "vanilla"
    wandb.config.dim = 3
    wandb.config.n_forward_per_backwards = 1
    wandb.config.augmentation_noise_std = 0.001
    wandb.config.inner_fold = 1
    wandb.config.test_center = "umcg"

    train()

    wandb.finish()

# def infer(model, datamodule):
#     pass


# def leave_one_center_out_cross_validate(model, datamodule):
#     pass
# def leave_one_center_out_cross_validate(model, datamodule):
#     pass
