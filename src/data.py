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
from src.transforms import RandTranspose
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    ToTensord,
    RandFlipd,
)


class DataModule(LightningDataModule):
    def __init__(self, input_data_root, prediction_target_file_path, config):
        super().__init__()
        if config.method == "crop":
            folder_name = f"dim-{config.dim}_size-{config.size}_method-{config.method}_roi_size-{config.roi_size}"
        elif config.method == "zoom":
            folder_name = f"dim-{config.dim}_size-{config.size}_method-{config.method}_margin-{config.margin}"
        self.root = Path(input_data_root) / folder_name

        self.target = pd.read_csv(prediction_target_file_path).set_index("lesion")[
            config.lesion_target
        ]
        self.test_center = config.test_center
        self.max_batch_size = config.max_batch_size
        self.dim = config.dim

        self.train_transform = self.get_transform(augmented=True)
        self.val_transform = self.get_transform(augmented=True)
        self.test_transform = self.get_transform(augmented=False)

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
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)

    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_sampler=GroupedSampler(
                groups=[x["patient"] for x in self.test_dataset],
                shuffle=shuffle,
                max_batch_size=self.max_batch_size,
            ),
        )

    def get_transform(self, augmented=False):
        load = Compose(
            [
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
            ]
        )

        if self.dim == 3:
            augmentation = Compose(
                [
                    RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["img"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["img"], prob=0.5, spatial_axis=2),
                    RandTranspose(),
                ]
            )
        elif self.dim == 2:
            augmentation = Compose(
                [
                    RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
                    RandRotate90d(keys=["img"], prob=1, k=4),
                ]
            )

        if augmented:
            return Compose([load, augmentation, ToTensord(keys=["img"])])
        else:
            return Compose([load, ToTensord(keys=["img"])])


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
