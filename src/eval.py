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
from data import DataModule
from config import radiomics_folder, lesion_level_labels_csv

class TrainedModel(Module):
    def __init__(self, checkpoint_file):
        super().__init__()
        self.model = nets.SEResNet50(
            spatial_dims=3,
            in_channels=1,
            num_classes=1
        )

        state_dict = torch.load(checkpoint_file)['state_dict']
        formatted_state_dict = self.format_state_dict(state_dict)

        self.model.load_state_dict(formatted_state_dict)

        self.model.eval()

    def format_state_dict(self, state_dict):
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key[6:]] = state_dict[key]
            state_dict.pop(key)
        
        return state_dict

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class Ensemble(pl.LightningModule):
    def __init__(self, run_ids, predictions_save_file_path):
        super().__init__()
        
        self.ensemble = []
        for run_id in run_ids:
            checkpoint_file = self.get_checkpoint_file(run_id)
            model = TrainedModel(checkpoint_file)
            self.ensemble.append(model)

        self.predictions_save_file_path = predictions_save_file_path
        self.predictions = pd.DataFrame(columns=[f'fold_{f}' for f in range(5)])

    def forward(self, x):
        out = torch.stack([model(x).squeeze(-1) for model in self.ensemble], -1)
        return out

    def test_step(self, batch, batch_idx):
        x = batch['img']
        y_hat = self(x)

        self.predictions.loc[
            Path(y_hat.meta['filename_or_obj']).name
        ] = y_hat.detach().numpy().squeeze()

    def test_epoch_end(self, results):
        self.predictions.to_csv(self.predictions_save_file_path)

    def get_checkpoint_file(self, run_id):
        checkpoint_root = Path('/mnt/hpc/rens/premium_dl_ct/src/debugging')

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
    r = Path('/mnt/hpc/rens/premium_dl_ct/src/wandb')
    sweep_folder = r / f'sweep-{sweep_id}'

    fold_vs_id = defaultdict(list)
    for config_file in sweep_folder.iterdir():
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        fold_vs_id[config['test_center']['value']].append(config_file.stem.split('-')[-1])

    return fold_vs_id

if __name__ == "__main__":
    sweep_id = '1gg58uu2'
    fold_vs_id = get_fold_vs_run_ids(sweep_id)
    
    save_folder = Path('/mnt/c/Users/user/data/results_dl')
    
    test_center = 'radboud'
    
    (save_folder / test_center).mkdir(exist_ok=True)

    run_ids = fold_vs_id[test_center]

    e = Ensemble(run_ids, save_folder / test_center / 'dl_preds.csv')

    mock_wandb_config = Config(
        roi_selection_method='crop',
        dim = 3,
        size=182,
        roi_size=142,
        test_center=test_center,
        max_batch_size=1,
        inner_fold=0,
        lesion_target='lesion_response',
        sampler='vanilla',
        augmentation_noise_std=0.001
    )

    dm = DataModule(radiomics_folder, lesion_level_labels_csv, mock_wandb_config)

    trainer = Trainer(gpus=1)
    trainer.test(e, dm)