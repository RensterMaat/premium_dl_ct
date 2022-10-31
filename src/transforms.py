import torch
import numpy as np
from pathlib import Path
from itertools import permutations
from monai.transforms import Transform, SpatialCrop, BorderPad, SaveImage, Resize


class FindCentroid(Transform):
    def __call__(self, data):
        seg = data["seg"]
        centroid = [int(np.mean(coor)) for coor in np.where(seg)][1:]
        data["centroid"] = centroid
        return data


class GetFixedROISize(Transform):
    def __init__(self, roi_size=100):
        super().__init__()
        self.size = roi_size

    def __call__(self, data):
        assert (data["img"].affine == data["seg"].affine).all()

        roi_size = (self.size / data["img"].affine.diag()[:-1]).abs().int()
        data["roi_size"] = np.array(roi_size)

        return data


class PadForCrop(Transform):
    def __call__(self, data):
        border_size = (np.ceil(data["roi_size"] / 2)).astype(int).tolist()
        padder = BorderPad(border_size, value=-1024)
        data["img"] = padder(data["img"])
        data["centroid"] += np.array(border_size)
        return data


class CropToCentroidWithSize(Transform):
    def __call__(self, data):
        cropper = SpatialCrop(roi_center=data["centroid"], roi_size=data["roi_size"])
        data["img"] = cropper(data["img"])
        data["img"].meta["spatial_shape"] = np.array(data["img"].shape[1:])
        data["img"].meta["affine"] = data["img"].meta["affine"] * np.eye(4)
        return data


class ResizedAndSetMetadata(Transform):
    def __init__(self, spatial_size):
        super().__init__()
        self.spatial_size = spatial_size

    def __call__(self, data):
        resizer = Resize(spatial_size=self.spatial_size)
        data["img"] = resizer(data["img"])

        return data


class OrthogonalSlices(Transform):
    def __call__(self, data):
        center_slices = (np.array(data["img"].shape)[1:] / 2).astype(int)

        sagittal = data["img"][0, center_slices[0]]
        coronal = data["img"][0, :, center_slices[1]]
        transverse = data["img"][0, :, :, center_slices[2]]

        data["img"] = torch.stack([sagittal, coronal, transverse])

        return data


class Save(Transform):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir

    def __call__(self, data):
        postfix = Path(data["seg"].meta["filename_or_obj"]).name.split(".")[0][-1:]
        saver = SaveImage(
            output_dir=self.output_dir,
            output_postfix=postfix,
            separate_folder=False,
            resample=False,
        )
        saver(data["img"])

        return data


class GetZoomedROI(Transform):
    def __init__(self, margin=10):
        super().__init__()
        self.margin = margin

    def __call__(self, data):
        where = torch.where(data["seg"])[1:]
        bbox_min_coordinates = np.array([axis.min() for axis in where])
        bbox_max_coordinates = np.array([axis.max() for axis in where])
        bbox_dim = bbox_max_coordinates - bbox_min_coordinates

        pix_size = data["seg"].affine.diag().numpy()[:-1]
        bbox_size = bbox_dim * pix_size

        roi_size = bbox_size.max() + self.margin

        roi_dim = roi_size / pix_size
        margin_to_add = (roi_dim - bbox_dim) / 2
        roi_min_coordinates = (bbox_min_coordinates - margin_to_add).astype(int)
        roi_max_coordinates = (bbox_max_coordinates + margin_to_add).astype(int)

        data["roi_start"] = roi_min_coordinates
        data["roi_end"] = roi_max_coordinates
        data["pix_margin"] = np.array(np.ceil(margin_to_add).astype(int)).tolist()

        return data


class PadForZoom(Transform):
    def __call__(self, data):
        padder = BorderPad(data["pix_margin"], value=-1024)
        data["img"] = padder(data["img"])
        data["roi_start"] = data["roi_start"] + data["pix_margin"]
        data["roi_end"] = data["roi_end"] + data["pix_margin"]

        return data


class CropToROI(Transform):
    def __call__(self, data):
        cropper = SpatialCrop(roi_start=data["roi_start"], roi_end=data["roi_end"])
        data["img"] = cropper(data["img"])
        data["img"].meta["spatial_shape"] = np.array(data["img"].shape[1:])
        data["img"].meta["affine"] = data["img"].meta["affine"] * np.eye(4)
        return data


class RandTranspose(Transform):
    def __call__(self, data):
        axis_orderings = list(permutations([0, 1, 2]))
        chosen_ordering = random.sample(axis_orderings, 1)[0]
        ordering_including_batch = [0] + [ax + 1 for ax in chosen_ordering]

        data["img"] = data["img"].permute(ordering_including_batch)

        return data
