import numpy as np
from pathlib import Path
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

        roi_size = (self.size / data["img"].affine.diag()[:-1]).int()
        data["roi_size"] = np.array(roi_size)

        return data


class Pad(Transform):
    def __call__(self, data):
        border_size = (np.ceil(data["roi_size"] / 2)).astype(int).tolist()
        padder = BorderPad(border_size, value=-1024)
        data["img"] = padder(data["img"])
        data["centroid"] += np.array(border_size)
        return data


class CropToROI(Transform):
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
