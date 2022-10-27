import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from model import Model


def train(model, datamodule, config):
    model = Model()

    seed_everything(0)

    logger = WandbLogger(
        name='hello1',
        project='project_skeleton_on_lung_lesions',
    )

    trainer = Trainer(
        max_epochs=3,
        gpus=1,
        deterministic=True,
        fast_dev_run=False,
        logger=logger
    )
            
    trainer.fit(model, dm)

    wandb.finish()


def infer(model, datamodule):
    pass


def leave_one_center_out_cross_validate(model, datamodule):
    pass
