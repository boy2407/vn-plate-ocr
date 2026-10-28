"""Version-pinned trainer that keeps the corruption hook when Mosaic closes."""

from ultralytics.data.augment import Albumentations
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from custom_alpr.yolo_vast.corruptions import QualityTransform


class QualityDataset(YOLODataset):
    def __init__(self, *args, quality_arm="control", quality_log_dir=None, **kwargs):
        self.quality_arm, self.quality_log_dir = quality_arm, quality_log_dir
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            # Remove implicit optional-package transforms in BOTH arms.
            transforms.transforms = [
                t for t in transforms.transforms if not isinstance(t, Albumentations)
            ]
            transforms.insert(
                len(transforms.transforms) - 1,
                QualityTransform(self.quality_arm, self.quality_log_dir),
            )
        return transforms


class ControlTrainer(DetectionTrainer):
    quality_arm = "control"

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return QualityDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls,
            stride=stride,
            pad=0.0,
            prefix="train: ",
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            quality_arm=self.quality_arm,
            quality_log_dir=self.save_dir / "quality_counts",
        )


class ModerateTrainer(ControlTrainer):
    quality_arm = "moderate"
