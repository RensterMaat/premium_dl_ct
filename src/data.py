import pandas as pd
import itertools
from pathlib import Path
from monai.data import CacheDataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    ToTensord,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        path_to_center_folders,
        prediction_target_file_path,
        target,
        test_center,
        batch_size=32,
    ):
        super().__init__()
        self.root = Path(path_to_center_folders)
        self.target = pd.read_csv(prediction_target_file_path).set_index("lesion")[
            target
        ]
        self.test_center = test_center
        self.batch_size = batch_size
        self.train_transform = Compose(
            [
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
                CenterSpatialCropd(keys=["img"], roi_size=(96, 96, 96)),
                ToTensord(keys=["img", "label"]),
            ]
        )
        self.val_transform = self.train_transform
        self.test_transform = self.train_transform
        self.centers = [c.name for c in self.root.iterdir()]

    def setup(self, *args, **kwargs):
        dev_centers = [c for c in self.centers if not c == self.test_center]

        # development data
        dev_data = list(
            itertools.chain(
                *[self.data_dir_to_dict(self.root / c) for c in dev_centers]
            )
        )
        train_data, val_data = train_test_split(dev_data, test_size=0.75)
        self.train_dataset = CacheDataset(train_data, self.train_transform)
        self.val_dataset = CacheDataset(val_data, self.val_transform)

        # test data
        test_data = self.data_dir_to_dict(self.root / self.test_center)
        self.test_dataset = CacheDataset(test_data, self.test_transform)

    def data_dir_to_dict(self, dir):
        return [
            {"img": str(lesion_path), "label": self.target.loc[lesion_path.name]}
            for lesion_path in dir.iterdir()
            if lesion_path.name in self.target.index
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )
