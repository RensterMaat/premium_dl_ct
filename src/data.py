import pandas as pd
import itertools
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Sampler
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
        train_data, val_data = self.grouped_train_val_split(dev_data, val_fraction=0.25)
        self.train_dataset = CacheDataset(train_data, self.train_transform)
        self.val_dataset = CacheDataset(val_data, self.val_transform)

        # test data
        test_data = self.data_dir_to_dict(self.root / self.test_center)
        self.test_dataset = CacheDataset(test_data, self.test_transform)

    def grouped_train_val_split(self, dev_data, val_fraction):
        all_patients = [x["patient"] for x in dev_data]
        random.shuffle(all_patients)

        split_ix = int(len(all_patients) * (1 - val_fraction))

        return dev_data[:split_ix], dev_data[split_ix:]

    def data_dir_to_dict(self, dir):
        return [
            {
                "img": str(lesion_path),
                "label": self.target.loc[lesion_path.name],
                "patient": lesion_path.name.split(".")[0][:-2],
            }
            for lesion_path in dir.iterdir()
            if lesion_path.name in self.target.index
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=GroupedSampler(
                groups=[x["patient"] for x in self.train_dataset],
                shuffle=True,
                max_batch_size=16,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=GroupedSampler(
                groups=[x["patient"] for x in self.val_dataset],
                shuffle=False,
                max_batch_size=16,
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_sampler=GroupedSampler(
                groups=[x["patient"] for x in self.test_dataset],
                shuffle=False,
                max_batch_size=16,
            ),
        )


class GroupedSampler(Sampler):
    def __init__(self, groups, max_batch_size, shuffle=False):
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle

        self.unique_groups = np.unique(groups)

        self.indices_per_group = defaultdict(list)
        self.count_per_group = defaultdict(int)

        for ix, group in enumerate(groups):
            self.indices_per_group[group].append(ix)
            self.count_per_group[group] += 1

        self.make_batches_from_groups()

    def make_batches_from_groups(self):
        self.all_batches = []
        current_batch = []
        current_batch_length = 0
        for group in self.unique_groups:
            group_length = self.count_per_group[group]
            if current_batch_length + group_length > self.max_batch_size:
                self.all_batches.append(current_batch)
                current_batch = self.indices_per_group[group].copy()
                current_batch_length = group_length
            else:
                current_batch.extend(self.indices_per_group[group])
                current_batch_length += group_length
        if current_batch:
            self.all_batches.append(current_batch)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_batches)

        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)
