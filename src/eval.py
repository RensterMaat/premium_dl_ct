import yaml
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from collections import defaultdict
from torch.nn import Module
from monai.networks import nets
from pathlib import Path
from dataclasses import dataclass
from data import DataModule, CENTERS
from config import radiomics_folder, lesion_level_labels_csv


class TrainedModel(pl.LightningModule):
    def __init__(self, run_id, predictions_save_file_path, fold):
        super().__init__()
        self.model = nets.SEResNet50(
            spatial_dims=3,
            in_channels=1,
            num_classes=1
        )

        checkpoint_file = self.get_checkpoint_file(run_id)

        state_dict = torch.load(checkpoint_file)['state_dict']
        formatted_state_dict = self.format_state_dict(state_dict)

        self.model.load_state_dict(formatted_state_dict)

        self.model.eval()

        self.predictions_save_file_path = Path(predictions_save_file_path)
        self.predictions = pd.Series(name=f'fold_{fold}')

    def format_state_dict(self, state_dict):
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key[6:]] = state_dict[key]
            state_dict.pop(key)
        
        return state_dict

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x = batch['img']
        y_hat = torch.sigmoid(self(x)).cpu().detach().numpy().squeeze(-1)
        filenames = batch['img_meta_dict']['filename_or_obj']

        for filename, prediction in zip(filenames, y_hat):
            self.predictions.loc[Path(filename).name] = prediction

    def test_epoch_end(self, *args, **kwargs):
        if self.predictions_save_file_path.exists():
            other_folds = pd.read_csv(self.predictions_save_file_path).set_index('Unnamed: 0')
            combined_dataframe = other_folds.join(
                self.predictions.to_frame(),
                how='outer'
            )
            combined_dataframe.to_csv(self.predictions_save_file_path)
        else:
            self.predictions.to_frame().to_csv(self.predictions_save_file_path)

    def get_checkpoint_file(self, run_id):
        checkpoint_root = Path('/mnt/c/Users/user/data/models/debugging')

        checkpoint_folder = list(checkpoint_root.glob(f'{run_id}'))[0] / 'checkpoints'
        checkpoint_file = list(checkpoint_folder.glob('*.ckpt'))[0]

        return checkpoint_file


@dataclass
class Config:
    roi_selection_method: str
    dim: int
    size: int
    roi_size: int
    test_center: str
    max_batch_size: int
    inner_fold: int
    lesion_target: str
    sampler: str
    augmentation_noise_std: float


def get_fold_vs_run_ids(sweep_id):
    r = Path('/mnt/c/Users/user/data/models/wandb')
    sweep_folder = r / f'sweep-{sweep_id}'

    fold_vs_id = defaultdict(list)
    for config_file in sweep_folder.iterdir():
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        fold_vs_id[config['test_center']['value']].append(
            (config['inner_fold']['value'], config_file.stem.split('-')[-1])
        )

    return fold_vs_id

if __name__ == "__main__":
    sweep_id = '1gg58uu2'
    fold_vs_id = get_fold_vs_run_ids(sweep_id)
    
    save_folder = Path('/mnt/c/Users/user/data/results_dl')

    trainer = Trainer(gpus=1)

    mock_wandb_config = Config(
        roi_selection_method='crop',
        dim = 3,
        size=182,
        roi_size=142,
        test_center=None,
        max_batch_size=24,
        inner_fold=None,
        lesion_target='lesion_response',
        sampler='vanilla',
        augmentation_noise_std=0.001
    )
    
    for test_center in CENTERS:
        (save_folder / test_center).mkdir(exist_ok=True)

        for fold, run_id in fold_vs_id[test_center]:
            mock_wandb_config.inner_fold = fold
            mock_wandb_config.test_center = test_center
            dm = DataModule(radiomics_folder, lesion_level_labels_csv, mock_wandb_config)
            dm.setup()

            # inference on validation fold for later recalibration and combination model
            model = TrainedModel(
                run_id, save_folder / test_center / 'dl_training_preds.csv', 
                fold
            )

            trainer.test(model, dataloaders=dm.val_dataloader())

            # inference on test set
            model.predictions_save_file_path = save_folder / test_center / 'dl_test_preds.csv'

            trainer.test(model, dataloaders=dm.test_dataloader())