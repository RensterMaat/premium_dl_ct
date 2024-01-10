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
from transforms import RandMirror
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    CenterSpatialCropd,
    RandGaussianNoised,
)


CENTERS = [
    "amphia",
    "isala",
    "lumc",
    "maxima",
    "mst",
    "radboud",
    "umcg",
    "umcu",
    "vumc",
    "zuyderland",
]


class DataModule(LightningDataModule):
    def __init__(self, input_data_root, prediction_target_file_path, config):
        super().__init__()
        if config.roi_selection_method == "crop":
            folder_name = f"dim-{config.dim}_size-{config.size}_method-{config.roi_selection_method}_roi_size-{config.roi_size}"
        elif config.roi_selection_method == "zoom":
            folder_name = f"dim-{config.dim}_size-{config.size}_method-{config.roi_selection_method}_margin-{config.margin}"
        self.root = Path(input_data_root) / folder_name

        self.lesion_level_data = pd.read_csv(
            prediction_target_file_path, sep=";"
        ).set_index("lesion")
        self.test_center = config.test_center
        self.max_batch_size = config.max_batch_size
        self.dim = config.dim

        self.config = config

        self.train_transform = self.get_transform(augmented=True)
        self.val_transform = self.get_transform(augmented=True)
        self.test_transform = self.get_transform(augmented=False)

        self.centers = [c.name for c in self.root.iterdir()]

    def setup(self, stage=None):
        dev_centers = [c for c in self.centers if not c == self.test_center]

        # development data
        if stage == "fit" or stage is None:
            dev_data = list(
                itertools.chain(
                    *[self.data_dir_to_dict(self.root / c) for c in dev_centers]
                )
            )

            self.train_data, self.val_data = self.grouped_train_val_split(
                dev_data, fold=self.config.inner_fold
            )

        # test data
        if stage == "test" and self.test_center:
            self.test_data = self.data_dir_to_dict(self.root / self.test_center)

    def grouped_train_val_split(self, dev_data, fold):
        all_patients = np.unique([x["patient"] for x in dev_data])
        random.shuffle(all_patients)

        patient_vs_fold = self.lesion_level_data.groupby("patient").fold.first()

        train_data = [
            x
            for x in dev_data
            if patient_vs_fold[x["patient"]] != fold and not np.isnan(x["label"])
        ]
        # train_data = []
        # for x in dev_data:
        #     print(x)
        #     print(patient_vs_fold[x["patient"]], x["label"])
        #     print()
        #     if patient_vs_fold[x["patient"]] != fold and not np.isnan(x["label"]):
        #         train_data.append(x)

        val_data = [x for x in dev_data if patient_vs_fold[x["patient"]] == fold]

        return train_data, val_data

    def data_dir_to_dict(self, dir):
        return [
            {
                "img": str(lesion_path),
                "label": self.lesion_level_data.loc[
                    lesion_path.name, self.config.lesion_target
                ],
                "patient": lesion_path.name.split(".")[0][:-2]
                .replace("abdomen", "")
                .replace("thorax", "")
                .replace("hals", ""),
                "organ": self.lesion_level_data.loc[lesion_path.name, "organ"],
            }
            for lesion_path in dir.iterdir()
            if lesion_path.name in self.lesion_level_data.index
        ]

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_data, transform=self.train_transform, shuffle=True
        )

    def val_dataloader(self):
        return self.get_dataloader(self.val_data, transform=self.val_transform)

    def test_dataloader(self):
        return self.get_dataloader(self.test_data, transform=self.test_transform)

    def get_dataloader(self, data, transform, shuffle=False):
        dataset = CacheDataset(data, transform)
        batch_sampler = self.get_sampler(data, shuffle)

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            batch_size=self.config.max_batch_size if not batch_sampler else 1,
            num_workers=12,
        )

    def get_sampler(self, data, shuffle):
        if self.config.sampler == "patient_grouped":
            return GroupedSampler(
                groups=[x["patient"] for x in data],
                shuffle=shuffle,
                max_batch_size=self.max_batch_size,
            )
        elif self.config.sampler == "label_stratified":
            return StratifiedSampler(
                groups=[[x["label"]] for x in data],
                shuffle=shuffle,
                batch_size=self.max_batch_size,
            )
        elif self.config.sampler == "label_organ_stratified":
            return StratifiedSampler(
                groups=[[x["label"], x["organ"]] for x in data],
                shuffle=shuffle,
                batch_size=self.max_batch_size,
            )
        elif self.config.sampler == "vanilla":
            return None

    def get_transform(self, augmented=False):
        load = Compose(
            [
                LoadImaged(keys=["img"]),
            ]
        )

        if self.dim == 3:
            augmentation = Compose(
                [
                    EnsureChannelFirstd(keys=["img"]),
                    RandMirror(prob=0.5, spatial_axis=0),
                    RandRotated(
                        keys=["img"],
                        range_x=(0, 2 * np.pi),
                        range_y=(0, 2 * np.pi),
                        range_z=(0, 2 * np.pi),
                        prob=1,
                    ),
                    CenterSpatialCropd(keys=["img"], roi_size=(128, 128, 128)),
                    RandGaussianNoised(
                        keys=["img"], prob=1, std=self.config.augmentation_noise_std
                    ),
                ]
            )
        elif self.dim == 2:
            augmentation = Compose(
                [
                    RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),  # checken !!!!!
                    RandRotate90d(keys=["img"], prob=1, max_k=4),
                ]
            )

        if augmented:
            return Compose([load, augmentation, ToTensord(keys=["img"])])
        else:
            return Compose(
                [load, EnsureChannelFirstd(keys=["img"]), ToTensord(keys=["img"])]
            )


class StratifiedSampler(Sampler):
    def __init__(self, groups, batch_size, shuffle=False):
        self.groups = np.array(["_".join([str(el) for el in x]) for x in groups])
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.make_batches_from_labels()

    def make_batches_from_labels(self):
        last_batch_size = len(self.groups) % self.batch_size

        if last_batch_size > 0:
            last_batch = list(range(len(self.groups)))[-last_batch_size:]
            groups_of_ordinary_batches = self.groups[:-last_batch_size]
        else:
            last_batch = []
            groups_of_ordinary_batches = self.groups

        unique_groups = np.unique(groups_of_ordinary_batches)

        group_indices = []
        for unique_group in unique_groups:
            indices = np.where(groups_of_ordinary_batches == unique_group)[0]
            group_indices.append(indices)

        ordinary_batches = (
            np.concatenate(group_indices)
            .reshape(self.batch_size, -1)
            .transpose()
            .tolist()
        )

        self.all_batches = ordinary_batches

        if last_batch:
            self.all_batches.append(last_batch)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_batches)

        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)


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
