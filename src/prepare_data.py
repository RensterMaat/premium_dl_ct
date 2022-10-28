import numpy as np
from pathlib import Path

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
)
from transforms import (
    FindCentroid,
    GetFixedROISize,
    CropToROI,
    Pad,
    Save,
)


def extract_3d_roi(segmentation_path, destination_path, roi_size=100, save_size=96):
    pipeline = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            FindCentroid(),
            GetFixedROISize(eventual_roi_size),
            Pad(),
            CropToROI(),
            Resized(keys=["img"], spatial_size=[save_size] * 3),
            ScaleIntensityRanged(
                keys=["img"], a_min=-1024, a_max=3000, b_min=0, b_max=1, clip=True
            ),
            Save(output_dir=destination_path),
        ]
    )

    segmentation_path = Path(segmentation_path)
    scan_path = find_corresponding_scan(segmentation_path)
    data = {"img": str(scan_path), "seg": str(segmentation_path)}

    pipeline(data)


def preprocess_dir(segmentation_dir, destination_dir):
    for segmentation_path in Path(segmentation_dir).iterdir():
        extract_3d_roi(segmentation_path, destination_dir)


def find_corresponding_scan(segmention_path):
    scan_folder = segmention_path.parent.parent / "scans"
    scan_name = segmention_path.name.split(".")[0][:-2] + ".nii.gz"

    return scan_folder / scan_name


preprocess_dir(
    r"D:\premium_data\amphia\monotherapy\split_segmentations",
    r"C:\Users\user\data\dl_radiomics\preprocessed_3d\amphia",
)

preprocess_dir(
    r"D:\premium_data\mst\monotherapy\split_segmentations",
    r"C:\Users\user\data\dl_radiomics\preprocessed_3d\mst",
)

preprocess_dir(
    r"D:\premium_data\zuyderland\monotherapy\split_segmentations",
    r"C:\Users\user\data\dl_radiomics\preprocessed_3d\zuyderland",
)
