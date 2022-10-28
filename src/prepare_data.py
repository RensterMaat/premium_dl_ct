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
    CropToCentroidWithSize,
    PadForCrop,
    Save,
    OrthogonalSlices,
    CropToROI,
    GetZoomedROI,
    PadForZoom,
)


class Preprocessor:
    def __init__(self, output_folder, output_dim, output_size):
        self.output_folder = output_folder
        self.output_size = output_size

        assert output_dim in [2, 3]
        self.output_dim = output_dim

    def make_pipeline(self):
        self.pipeline = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                self.roi_selection_pipeline,
                Resized(keys=["img"], spatial_size=[self.output_size] * 3),
                ScaleIntensityRanged(
                    keys=["img"], a_min=-1024, a_max=3000, b_min=0, b_max=1, clip=True
                ),
                self.get_dimensionality_selection(),
                Save(output_dir=self.output_root_folder),
            ]
        )

    def get_roi_selection_pipeline():
        pass

    def get_dimensionality_selection(self):
        if self.output_dim == 2:
            return OrthogonalSlices()
        else:
            return Identityd(keys=["img", "seg"])

    def __call__(self, data):
        return self.pipeline(data)


class CropPreprocessor(Preprocessor):
    def __init__(self, output_folder, output_dim, output_size, roi_size):
        super().__init__(output_folder, output_dim, output_size)

        self.roi_selection_pipeline = Compose(
            [
                FindCentroid(),
                GetFixedROISize(roi_size),
                CropPad(),
                CropToROI(),
            ]
        )


class ZoomPreprocessor(Preprocessor):
    def __init__(self, output_folder, output_dim, output_size, margin):
        super().__init__(output_folder, output_dim, output_size)
        self.margin = margin

        self.roi_selection_pipeline = Compose(
            [GetZoomedROI(margin), PadForZoom(), CropToROI()]
        )


def extract_3d_roi(segmentation_path, destination_path, roi_size=100, save_size=96):
    pipeline = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            FindCentroid(),
            GetFixedROISize(roi_size),
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
    scan_folder = Path(segmention_path).parent.parent / "scans"
    scan_name = Path(segmention_path).name.split(".")[0][:-2] + ".nii.gz"

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
