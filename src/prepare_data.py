import numpy as np
import os
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    Identityd,
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

        self.make_pipeline()

    def make_pipeline(self):
        self.pipeline = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                self.get_roi_selection_pipeline(),
                Resized(keys=["img"], spatial_size=[self.output_size] * 3),
                ScaleIntensityRanged(
                    keys=["img"], a_min=-1024, a_max=3000, b_min=0, b_max=1, clip=True
                ),
                self.get_dimensionality_selection(),
                Save(output_dir=self.output_folder),
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

    def set_output_folder(self, path):
        self.output_folder = path
        self.make_pipeline()


class CropPreprocessor(Preprocessor):
    def __init__(self, output_folder, output_dim, output_size, roi_size):
        self.roi_size = roi_size
        super().__init__(output_folder, output_dim, output_size)

    def get_roi_selection_pipeline(self):
        return Compose(
            [
                FindCentroid(),
                GetFixedROISize(self.roi_size),
                PadForCrop(),
                CropToCentroidWithSize(),
            ]
        )


class ZoomPreprocessor(Preprocessor):
    def __init__(self, output_folder, output_dim, output_size, margin):
        self.margin = margin
        super().__init__(output_folder, output_dim, output_size)

    def get_roi_selection_pipeline(self):
        return Compose([GetZoomedROI(self.margin), PadForZoom(), CropToROI()])


class DataPipeline:
    def __init__(
        self, input_folders, output_root, output_dim, output_size, method, **kwargs
    ):
        self.input_folders = input_folders

        output_folder_name = (
            f"dim-{output_dim}_size-{output_size}_method-{method}_"
            + "_".join([f"{k}-{v}" for (k, v) in kwargs.items()])
        )
        self.output_folder = Path(output_root) / output_folder_name
        os.makedirs(self.output_folder, exist_ok=True)

        if method == "crop":
            self.processor = CropPreprocessor(None, output_dim, output_size, **kwargs)
        elif method == "zoom":
            self.processor = ZoomPreprocessor(None, output_dim, output_size, **kwargs)

    def run(self):
        for input_folder in self.input_folders:
            input_folder = Path(input_folder)
            center = input_folder.parent.parent.name

            target_folder = self.output_folder / center
            os.makedirs(target_folder, exist_ok=True)

            self.processor.set_output_folder(target_folder)

            for segmentation_file in input_folder.iterdir():
                scan_file = self.find_corresponding_scan(segmentation_file)
                data = {"img": scan_file, "seg": segmentation_file}
                self.processor(data)

    def find_corresponding_scan(self, segmention_path):
        scan_folder = Path(segmention_path).parent.parent / "scans"
        scan_name = Path(segmention_path).name.split(".")[0][:-2] + ".nii.gz"

        return scan_folder / scan_name


def main():
    CENTERS = [
        "amphia",
        # "isala",
        # "lumc",
        "maxima",
        "mst",
        "radboud",
        "umcu",
        "vumc",
        "zuyderland",
    ]

    r = Path(r"D:\premium_data")
    input_folders = []
    for center in CENTERS:
        input_folders.extend(
            [
                r / center / "monotherapy" / "split_segmentations",
                r / center / "combination_therapy" / "split_segmentations",
            ]
        )

    # DataPipeline(
    #     input_folders,
    #     r"C:\Users\user\data\dl_radiomics",
    #     3,
    #     128,
    #     method="crop",
    #     roi_size=50,
    # ).run()

    # DataPipeline(
    #     input_folders,
    #     r"C:\Users\user\data\dl_radiomics",
    #     2,
    #     256,
    #     method="crop",
    #     roi_size=100,
    # ).run()

    # DataPipeline(
    #     input_folders,
    #     r"C:\Users\user\data\dl_radiomics",
    #     2,
    #     256,
    #     method="crop",
    #     roi_size=150,
    # ).run()

    DataPipeline(
        input_folders,
        r"C:\Users\user\data\dl_radiomics",
        2,
        256,
        method="zoom",
        margin=0,
    ).run()

    DataPipeline(
        input_folders,
        r"C:\Users\user\data\dl_radiomics",
        2,
        256,
        method="zoom",
        margin=10,
    ).run()

    DataPipeline(
        input_folders,
        r"C:\Users\user\data\dl_radiomics",
        2,
        256,
        method="zoom",
        margin=50,
    ).run()


if __name__ == "__main__":
    main()
